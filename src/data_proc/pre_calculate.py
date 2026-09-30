import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit"


class LLM:
    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id

    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # o "fp4"
            bnb_4bit_compute_dtype=torch.bfloat16,  # o torch.float16 si no tienes soporte bf16
        )
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_config)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.embeddings = self.model.get_input_embeddings()

    def unload_model(self):
        del self.model
        # del self.tokenizer
        self._cleanup()

    def _cleanup(self):
        torch.cuda.empty_cache()
        while gc.collect() != 0:
            break

    def run(self, label):
        input_ids = self.tokenizer(label, return_tensors="pt").input_ids.to('cuda')
        self.embeddings = self.embeddings.to('cuda')
        t = self.embeddings(input_ids[0])
        return t
