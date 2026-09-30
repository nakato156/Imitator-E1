"""GemmaBridge: frozen Gemma-3n LLM wrapper for the v116 soft-prefix pipeline.

The bridge encapsulates the LLM so that the rest of the training code never
touches the HuggingFace model directly.  It is instantiated once in
Trainer.__init__ when loss_type == "ce_ar" and stays on GPU for the entire run.

Forward path (ce_ar loss):
    keypoints → Imitator → PrefixAdapter → prefix [B, K, hidden_size]
                                              |
              concat with embed_tokens(target_ids[:, :-1])
                                              |
                                  inputs_embeds [B, K+L-1, hidden_size]
                                              |
                        GemmaBridge.forward(inputs_embeds, attn_mask)
                                              |
                                  logits [B, K+L-1, vocab_size]

Loading note (Gemma-3n / unsloth checkpoints):
    Use unsloth.FastModel.from_pretrained to correctly initialize BNB 4-bit
    quantization state.  Loading with plain AutoModelForCausalLM leaves
    LinearFP4 layers in an uninitialized state that causes AssertionError on
    the first forward pass.

    Embedding space: GemmaScaledWordEmbedding multiplies weights by
    sqrt(hidden_size) = 45.25 for E2B.  embed_tokens() returns scaled
    embeddings — the prefix adapter must produce vectors in this same scale.

    Gradient flow: inputs_embeds path supports autograd (grad_fn is preserved)
    even when all LLM weights are frozen (requires_grad=False).  Gradients
    flow back to the prefix via inputs_embeds without accumulating in LLM
    weights.
"""

import os
from types import SimpleNamespace

import torch
import torch.nn as nn


def _load_lm(model_id: str, max_seq_length: int = 128):
    """Load LLM via unsloth.FastModel for correct 4-bit quantization init.

    For multimodal models (Gemma3nForConditionalGeneration), extracts only
    the text language_model component and frees the vision/audio encoders from
    CUDA memory — those are not needed for soft-prefix training.

    Falls back to standard HuggingFace loading if unsloth is not available.
    Returns (model, tokenizer) — the tokenizer is needed for generation decoding.
    """
    import gc

    # Unsloth can decorate internal Gemma-3n kernels with torch.compile even
    # in inference mode.  Some supported environments intentionally pin a
    # Torch/Triton pair without the private ``triton_key`` API, so compilation
    # fails before the first real forward.  The bridge favors portable eager
    # execution over compilation.
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        torch._dynamo.config.disable = True
    except Exception:
        pass

    try:
        from unsloth import FastModel
        full_model, tokenizer = FastModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        # Put in pure inference mode — prevents unsloth from applying
        # torch.compile / auto-compiling the forward, which would cache
        # dequantized NF4 weights in CUDA and use 5-6 GB of extra VRAM.
        try:
            FastModel.for_inference(full_model)
            print(f"[GemmaBridge] FastModel.for_inference() aplicado: no auto-compile")
        except Exception as e_inf:
            print(f"[GemmaBridge] for_inference() falló ({type(e_inf).__name__}): {e_inf}")
        # Also disable dynamo graph capture on the model to stay in eager mode.
        try:
            import torch._dynamo as dynamo
            dynamo.disable(full_model)
        except Exception:
            pass
        full_model_type = type(full_model).__name__
        # Multimodal wrapper: try to extract text-only sub-model to free
        # vision/audio encoders from CUDA (Gemma3nForConditionalGeneration
        # may or may not expose a language_model attribute depending on version).
        if hasattr(full_model, "language_model"):
            text_model = full_model.language_model
            del full_model
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[GemmaBridge] Extraído language_model de {full_model_type}: {type(text_model).__name__}")
            return text_model, tokenizer
        print(f"[GemmaBridge] Cargado con unsloth.FastModel: {full_model_type}")
        return full_model, tokenizer
    except ImportError:
        print("[GemmaBridge] unsloth no disponible; usando HF estándar (puede fallar con bnb-4bit)")
    except Exception as e_unsloth:
        print(f"[GemmaBridge] unsloth.FastModel falló ({type(e_unsloth).__name__}): {e_unsloth}; usando HF estándar")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
    except Exception as e1:
        print(f"[GemmaBridge] AutoModelForCausalLM falló ({type(e1).__name__}); probando ImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


class GemmaBridge(nn.Module):
    """Frozen Gemma-3n wrapper for soft-prefix autoregressive training (v116).

    Parameters
    ----------
    model_id:
        HuggingFace model id or local path (e.g. ``"google/gemma-3n-E2B-it"``).
    device:
        Target device string.  ``device_map="cuda"`` is passed to
        ``from_pretrained``; this parameter is kept for API symmetry.

    Attributes
    ----------
    hidden_size : int
        Embedding / hidden dimension of the LLM (e.g. 2048 for gemma-3n-E2B).
    vocab_size : int
        Size of the token vocabulary.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        super().__init__()

        # Load text model and tokenizer (tokenizer needed for decode()).
        lm, self.tokenizer = _load_lm(model_id)
        lm.requires_grad_(False)
        lm.eval()
        # Store as a non-parameter attribute so that PyTorch doesn't try to
        # move it again when the bridge itself is moved (it already lives on
        # CUDA via device_map).
        self.lm = lm

        # _load_lm already extracts language_model for multimodal checkpoints,
        # so lm is always the text-only model here.  The routing below handles
        # the rare case where a HF fallback returns a full multimodal model.
        if hasattr(lm, "language_model"):
            self._text_model = lm.language_model
        else:
            self._text_model = lm
        self._language_model = None
        self._lm_head = None
        if hasattr(lm, "model") and hasattr(lm.model, "language_model") and hasattr(lm, "lm_head"):
            self._language_model = lm.model.language_model
            self._lm_head = lm.lm_head

        # Expose integer attributes for downstream modules.
        embed_layer = lm.get_input_embeddings()
        # embed_layer.weight: [vocab_size, hidden_size]
        self.vocab_size: int = embed_layer.weight.shape[0]
        self.hidden_size: int = embed_layer.weight.shape[1]
        # Cache the embedding layer to avoid re-fetching on every embed_tokens call.
        self._embed_layer = embed_layer

        # Pre-build an eager (non-compiled) forward wrapper so that
        # torch._dynamo.disable is applied once at construction time rather
        # than on every forward call.  This prevents any surrounding
        # torch.compile region from tracing into the Gemma model and caching
        # large dequantized NF4 weight buffers in CUDA memory.
        _text = self._text_model
        _language_model = self._language_model
        _lm_head = self._lm_head
        _config = getattr(lm.config, "text_config", lm.config)
        _final_logit_softcapping = getattr(_config, "final_logit_softcapping", None)

        @torch._dynamo.disable
        def _lm_forward(**kwargs):
            if kwargs.get("per_layer_inputs", None) is not None:
                if _language_model is None or _lm_head is None:
                    raise ValueError("per_layer_inputs requiere acceso a language_model + lm_head")
                outputs = _language_model(
                    input_ids=None,
                    per_layer_inputs=kwargs.pop("per_layer_inputs"),
                    attention_mask=kwargs.get("attention_mask"),
                    position_ids=kwargs.get("position_ids"),
                    past_key_values=kwargs.get("past_key_values"),
                    inputs_embeds=kwargs.get("inputs_embeds"),
                    use_cache=kwargs.get("use_cache"),
                    return_dict=True,
                )
                logits = _lm_head(outputs.last_hidden_state)
                if _final_logit_softcapping is not None:
                    logits = logits / _final_logit_softcapping
                    logits = torch.tanh(logits)
                    logits = logits * _final_logit_softcapping
                return SimpleNamespace(
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=getattr(outputs, "hidden_states", None),
                    attentions=getattr(outputs, "attentions", None),
                )
            return _text(**kwargs)

        self._lm_forward = _lm_forward

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids using the LLM's input embedding layer.

        The layer is ``GemmaScaledWordEmbedding``, which already applies the
        ``sqrt(hidden_size)`` scaling factor internally — do NOT scale again.

        Parameters
        ----------
        ids:
            Long tensor of shape ``[B, L]``.

        Returns
        -------
        torch.Tensor
            Float tensor of shape ``[B, L, hidden_size]``.
        """
        with torch.no_grad():
            return self._embed_layer(ids)

    def get_per_layer_inputs(self, ids: torch.Tensor) -> torch.Tensor:
        """Return Gemma-3n per-layer token embeddings for discrete IDs.

        Gemma-3n text uses an auxiliary per-layer embedding (PLE) table.  The
        full conditional-generation wrapper masks IDs outside the PLE vocab to
        zero before lookup; this helper mirrors that behavior.
        """
        text = self._text_model
        if hasattr(text, "model") and hasattr(text.model, "language_model"):
            text = text.model.language_model
        elif hasattr(text, "language_model"):
            text = text.language_model
        elif hasattr(text, "model") and hasattr(text.model, "get_per_layer_inputs"):
            text = text.model

        if not hasattr(text, "get_per_layer_inputs"):
            raise AttributeError("el modelo cargado no expone get_per_layer_inputs")

        limit = getattr(text.config, "vocab_size_per_layer_input", None)
        if limit is not None:
            ids = torch.where(
                (ids >= 0) & (ids < limit),
                ids,
                torch.zeros_like(ids),
            )
        with torch.no_grad():
            return text.get_per_layer_inputs(ids)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run a text-only forward pass through the LLM.

        Parameters
        ----------
        inputs_embeds:
            Float tensor of shape ``[B, S, hidden_size]``.  ``S`` is the
            total sequence length (prefix tokens + text tokens - 1).
        attention_mask:
            Bool / long tensor of shape ``[B, S]``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``[B, S, vocab_size]``.
        """
        # Cast to the LLM weight dtype (always bf16 for quantized Gemma).
        # PrefixImitator output can be float32 (F.normalize stays in float32
        # during autocast), which would cause dtype mismatch in Gemma's linears.
        target_dtype = self._embed_layer.weight.dtype
        inputs_embeds = inputs_embeds.to(dtype=target_dtype)
        out = self._lm_forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.logits

    @torch.no_grad()
    def _generate_greedy(
        self,
        *,
        input_ids: "torch.Tensor | None" = None,
        inputs_embeds: "torch.Tensor | None" = None,
        per_layer_inputs: "torch.Tensor | None" = None,
        attention_mask: "torch.Tensor | None" = None,
        max_new_tokens: int = 50,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("proporcione exactamente uno de input_ids o inputs_embeds")

        prefix = input_ids if input_ids is not None else inputs_embeds
        assert prefix is not None
        batch, seq_len = prefix.shape[:2]
        device = prefix.device
        if attention_mask is None:
            attention_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)

        kwargs = {
            "attention_mask": attention_mask,
            "use_cache": True,
        }
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        else:
            kwargs["inputs_embeds"] = inputs_embeds.to(dtype=self._embed_layer.weight.dtype)
            if per_layer_inputs is not None:
                kwargs["per_layer_inputs"] = per_layer_inputs.to(dtype=self._embed_layer.weight.dtype)

        generated = []
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        eos_id = self.tokenizer.eos_token_id
        out = self._lm_forward(**kwargs)

        for step in range(max_new_tokens):
            next_id = out.logits[:, -1, :].argmax(dim=-1)
            if eos_id is not None:
                next_id = torch.where(finished, torch.full_like(next_id, eos_id), next_id)
                finished |= next_id.eq(eos_id)
            generated.append(next_id)
            if eos_id is not None and bool(finished.all()):
                break
            if step + 1 == max_new_tokens:
                break

            attention_mask = torch.cat(
                [attention_mask, torch.ones(batch, 1, dtype=attention_mask.dtype, device=device)],
                dim=1,
            )
            out = self._lm_forward(
                input_ids=next_id.unsqueeze(1),
                attention_mask=attention_mask,
                past_key_values=out.past_key_values,
                use_cache=True,
            )

        if not generated:
            return torch.empty(batch, 0, dtype=torch.long, device=device)
        return torch.stack(generated, dim=1)

    def generate_from_ids(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        attention_mask: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Greedy decode using Gemma's native discrete-token path."""
        return self._generate_greedy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,
        max_new_tokens: int = 50,
        attention_mask: "torch.Tensor | None" = None,
        per_layer_inputs: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Greedy autoregressive decode conditioned on a soft prefix.

        Parameters
        ----------
        prefix:
            Float tensor of shape ``[B, K, hidden_size]`` — the soft prefix
            produced by PrefixImitator.
        max_new_tokens:
            Number of tokens to generate greedily.
        attention_mask:
            Optional ``[B, K]`` long tensor (1 = attend, 0 = ignore).  Defaults
            to all-ones (attend all prefix positions).

        Returns
        -------
        torch.Tensor
            Long tensor of shape ``[B, max_new_tokens]`` — generated token IDs.
        """
        return self._generate_greedy(
            inputs_embeds=prefix,
            per_layer_inputs=per_layer_inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

    def decode(self, token_ids: torch.Tensor) -> "list[str]":
        """Decode token IDs to strings using the LLM's tokenizer.

        Parameters
        ----------
        token_ids:
            Long tensor of shape ``[B, L]`` or ``[L]``.

        Returns
        -------
        list[str]
            One decoded string per batch element.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        return self.tokenizer.batch_decode(token_ids.cpu(), skip_special_tokens=True)
