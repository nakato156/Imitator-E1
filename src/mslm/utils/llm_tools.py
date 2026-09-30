import torch
from unsloth import FastLanguageModel
import torch.nn.functional as F
import gc

class Tools:
    def __init__(self, dtype=None, max_seq_length:int=4096, load_in_4bit:bool=True):
        model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-3B-Instruct",
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        # Obtener capa de embeddings
        self.embedding_layer = model.get_input_embeddings()

        # obtener lm_head
        self.lm_head = model.lm_head
        del model
        self._cleanup()

    def getLLM(self):
        return self.embedding_layer, self.tokenizer

    def _cleanup(self):
        torch.cuda.empty_cache()
        while gc.collect() != 0:
            break
    
    def get_logits_from_embedding(self, embeddings) -> str:
        logits = embeddings @ self.embedding_layer.weight.T
        return F.log_softmax(logits, dim=-1)
