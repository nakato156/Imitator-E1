import torch
import torch.nn.functional as F

class MultimodalSignLM:
    def __init__(self, base_model, tokenizer, device):
        """
        Initialize the MultimodalSignLM class.
        Params
        :model: LLama 3 model.
        :tokenizer: The tokenizer for the model.
        :device: The device to run the model on (e.g., 'cuda' or 'cpu').
        """

        self.model = base_model
        self.tokenizer = tokenizer
        self.device = device

        # Get the embeddings of all tokens in the vocabulary
        self.all_embeddings = base_model.get_input_embeddings().weight.data.to(self.device)

    def process_inputs(self, keypoints_embeddings, text_input:str):
        # Preprocess the inputs
        keypoints_embeddings = keypoints_embeddings.to(self.device)
        inputs = self.tokenizer(text_input, return_tensors="pt").to(self.device)
        
        sign_embed_tokens = torch.tensor(
            [self._find_closest_token(emb, self.all_embeddings) for emb in keypoints_embeddings[0]]  # embeddings[0] porque es un batch de tamaño 1
        ).unsqueeze(0).to(self.device)

        # Add EOS token at the end of the keypoints
        eos_token_id = self.tokenizer.eos_token_id
        sign_embed_tokens = torch.cat([sign_embed_tokens, torch.tensor([[eos_token_id]]).to(self.device)], dim=1)

        # Concatenate the token embeddings with the text input
        inputs['input_ids'] = torch.cat([inputs['input_ids'], sign_embed_tokens], dim=1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(sign_embed_tokens)], dim=1)
        
        return inputs
    
    def generate(self, keypoints_embeddings, text_input:str):
        self.model.eval()

        chat_format = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 07 Apr 2025\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        # Process the inputs
        inputs = self.process_inputs(keypoints_embeddings, chat_format + text_input)

        # Generate the output
        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
            )

        # Decode the output
        decoded_output = self.tokenizer.decode(output[0], skip_special_tokens=False)
        return decoded_output
    
    def _find_closest_token(self, embedding, all_embeddings):
        embedding = embedding.to(self.device)
        if embedding.dim() > 1:
            embedding = embedding.squeeze()

        # Calcular similitud del coseno
        similarities = F.cosine_similarity(embedding.unsqueeze(0), all_embeddings, dim=1)

        # Encontrar el índice del token más similar
        closest_token_id = torch.argmax(similarities).item()
        return closest_token_id

    def embeddings_to_text(self, embeddings: torch.Tensor) -> str:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.eval()

        embeddings = embeddings.to(device)

        embedding_matrix = self.all_embeddings.to(device)  # [V, D]

        embedding_matrix_norm = F.normalize(embedding_matrix, p=2, dim=1)  # [V, D]
        print(embedding_matrix_norm.shape)

        embeddings_norm = F.normalize(embeddings, p=2, dim=1)  # [T, D]

        similarities = torch.matmul(embeddings_norm, embedding_matrix_norm.T)  # [T, V]

        token_ids = torch.argmax(similarities, dim=1).tolist()
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)