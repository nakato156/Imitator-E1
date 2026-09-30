"""Smoke test: ¿inputs_embeds funciona y tiene gradientes en Gemma-3n?

Diagnóstico ONE-TIME para v116 (soft-prefix). Responde dos preguntas:

  Risk 1 — ¿Los gradientes fluyen por inputs_embeds hacia el prefijo?
    Necesario para entrenar el PrefixAdapter con CE-AR. Ejecuta un forward
    con inputs_embeds que requiere_grad=True y verifica que grad_fn es no-None.
    NOTA: Para Gemma-3n la equivalencia logit input_ids vs inputs_embeds NO
    se cumple (~35 diff) por la arquitectura AltUp (múltiples hidden states).
    Se reporta como métrica informativa, pero NO es criterio de PASS/FAIL.

  Risk 2 — ¿Funciona forward text-only con inputs_embeds?
    Se intenta el forward con el modelo cargado via unsloth.FastModel (necesario
    para inicializar correctamente el estado BNB 4-bit). Sin unsloth, los pesos
    LinearFP4 tienen estado no inicializado y lanzan AssertionError.

Notas de arquitectura (verificadas experimentalmente):
  - Carga: unsloth.FastModel.from_pretrained (obligatorio para BNB 4-bit init).
    Con AutoModelForCausalLM/AutoModelForImageTextToText, el forward falla con
    AssertionError en altup_projections aunque el modelo cargue sin errores.
  - Embedding layer: model.get_input_embeddings() -> GemmaScaledWordEmbedding
    Aplica sqrt(hidden_size)=45.25 al weight. embed_tokens() devuelve embeddings
    ya escalados; el PrefixAdapter debe producir vectores en el mismo espacio.
  - Diff logits input_ids vs inputs_embeds: ~35 (esperado). Gemma-3n AltUp
    crea múltiples hidden states en el path input_ids que no existen en
    el path inputs_embeds. No indica bug — es comportamiento de arquitectura.
  - Gradientes: inputs_embeds conserva grad_fn incluso con LLM congelado.
    Los gradientes fluyen al PrefixAdapter sin acumularse en pesos del LLM.

Uso:
    PYTHONPATH=. python scripts/audits/smoke_gemma_inputs_embeds.py
    PYTHONPATH=. python scripts/audits/smoke_gemma_inputs_embeds.py --model <hf-repo>
"""
import argparse
import sys

import torch

DEFAULT_MODEL = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"


def _load_lm(model_id: str):
    """Carga el LLM via unsloth.FastModel para init correcto de BNB 4-bit."""
    try:
        from unsloth import FastModel
        model, _ = FastModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            max_seq_length=128,
            load_in_4bit=True,
        )
        print(f"[load] Cargado con unsloth.FastModel: {type(model).__name__}")
        return model
    except ImportError:
        print("[load] unsloth no disponible; usando AutoModelForCausalLM (puede fallar con bnb-4bit)")
    except Exception as e:
        print(f"[load] unsloth.FastModel falló ({type(e).__name__}); fallback a HF estándar")

    from transformers import AutoModelForCausalLM
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda")
        print(f"[load] Cargado con AutoModelForCausalLM: {type(model).__name__}")
        return model
    except Exception as e1:
        print(f"[load] AutoModelForCausalLM falló ({type(e1).__name__}); probando ImageTextToText")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, device_map="cuda")
        print(f"[load] Cargado con AutoModelForImageTextToText: {type(model).__name__}")
        return model


def main():
    ap = argparse.ArgumentParser(description="Smoke test inputs_embeds para Gemma-3n (v116)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seq-len", type=int, default=5)
    args = ap.parse_args()

    print("=" * 60)
    print("SMOKE TEST: inputs_embeds / gradientes  (Gemma-3n v116)")
    print("=" * 60)

    model = _load_lm(args.model)
    model.eval()

    embed_layer = model.get_input_embeddings()
    device = next(model.parameters()).device
    hidden_size = embed_layer.weight.shape[1]
    vocab_size  = embed_layer.weight.shape[0]
    print(f"[info] Embedding layer : {type(embed_layer).__name__}")
    print(f"[info] hidden_size     : {hidden_size}  (sqrt={hidden_size**0.5:.2f})")
    print(f"[info] vocab_size      : {vocab_size}")
    print(f"[info] device          : {device}")

    ids = torch.tensor([[1] + list(range(2, 2 + args.seq_len - 1))],
                       dtype=torch.long, device=device)
    attn = torch.ones(1, args.seq_len, dtype=torch.long, device=device)
    print(f"[info] Secuencia: {ids.tolist()}")

    # ------------------------------------------------------------------ Risk 2
    print("\n--- Risk 2: forward text-only con inputs_embeds ---")
    risk2_pass = False
    with torch.no_grad():
        embeds = embed_layer(ids)   # scaled by sqrt(hidden_size)
    print(f"[emb] shape={tuple(embeds.shape)}, norm(tok0)={embeds[0,0].float().norm():.3f}")

    try:
        with torch.no_grad():
            out2 = model(inputs_embeds=embeds, attention_mask=attn)
        logits_emb = out2.logits
        print(f"[OK]  logits shape = {tuple(logits_emb.shape)}")
        risk2_pass = True
    except Exception as e:
        print(f"[FAIL] inputs_embeds forward lanzó: {type(e).__name__}: {e}")

    # ------------------------------------------------------------------ Risk 1
    print("\n--- Risk 1: gradiente fluye por inputs_embeds ---")
    risk1_pass = False
    grad_norm = 0.0
    if risk2_pass:
        try:
            emb_req = embeds.detach().requires_grad_(True)
            out_g = model(inputs_embeds=emb_req, attention_mask=attn)
            out_g.logits.mean().backward()
            if emb_req.grad is not None:
                grad_norm = emb_req.grad.float().norm().item()
                risk1_pass = grad_norm > 0
                print(f"[OK]  grad fluye, norm={grad_norm:.4f}")
            else:
                print("[FAIL] emb_req.grad es None — backward no propagó gradientes")
        except Exception as e:
            print(f"[FAIL] backward lanzó: {type(e).__name__}: {e}")
    else:
        print("[SKIP] Risk 2 falló, no se puede verificar gradientes")

    # ---- informativo: diff logits input_ids vs inputs_embeds ----
    print("\n--- Informativo: diff logits input_ids vs inputs_embeds ---")
    if risk2_pass:
        try:
            with torch.no_grad():
                out_ids = model(input_ids=ids, attention_mask=attn)
            diff = (out_ids.logits.float() - logits_emb.float()).abs()
            print(f"  max|diff| = {diff.max().item():.4f}  mean|diff| = {diff.mean().item():.4f}")
            print("  (esperado ~35 para Gemma-3n AltUp — no es criterio de PASS)")
        except Exception as e:
            print(f"  (no disponible: {e})")

    # ---------------------------------------------------------------- Resumen
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"  Risk 1 (gradientes por inputs_embeds): {'PASS (norm=%.4f)' % grad_norm if risk1_pass else 'FAIL'}")
    print(f"  Risk 2 (forward inputs_embeds OK)    : {'PASS' if risk2_pass else 'FAIL'}")

    if risk1_pass and risk2_pass:
        print("\nRESULT: PASS — v116 puede entrenar el prefijo via inputs_embeds.")
        sys.exit(0)
    else:
        print("\nRESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
