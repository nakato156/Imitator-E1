import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import TransformerEncoderLayerRoPE
from .components.stgcn import STGCNBlock, partition_adjacency
from torch.utils.checkpoint import checkpoint
import math

class Imitator(nn.Module):
    def __init__(
        self,
        A,  # raw adjacency matrix, no partitioning
        input_size: int,
        hidden_size: int = 512,
        output_size: int = 3072,
        nhead: int = 8,
        ff_dim: int = 1024,
        n_layers: int = 2,
        max_seq_length: int = 20, # cambiar
        encoder_dropout: int = 0.4,
        multihead_dropout: int = 0.1,
        pool_dim: int = 256,
    ):
        super().__init__()

        self.cfg = {
            "A": A,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "nhead": nhead,
            "ff_dim": ff_dim,
            "n_layers": n_layers,
            "max_seq_length": max_seq_length,
            "pool_dim": pool_dim,
            "encoder_dropout": encoder_dropout,
            "multihead_dropout": multihead_dropout,
        }

        print("Model Parameters: ", self.cfg)

        # --- Bloque de entrada ---
        A = partition_adjacency(A)
        self.stgcn = STGCNBlock(2, hidden_size // 2, A, kernel_size=3, stride=1)

        # Volvemos a hidden_size
        self.linear_hidden = nn.Sequential(
            nn.Conv2d(3 * (hidden_size // 2), hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size)
        )

        # Positional Encoding + Transformer
        encoder_layer    = TransformerEncoderLayerRoPE(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=encoder_dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.token_queries = nn.Parameter(torch.randn(max_seq_length, hidden_size))  # [1, hidden_size]
        # Queries = E_tokens [n_tokens × B × d], Keys/Values = frames_repr [T' × B × d]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=nhead,
            dropout=multihead_dropout,
            batch_first=True,
        )

        self.norm_attn = nn.LayerNorm(hidden_size)

        # Proyección final por paso de tiempo
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x:torch.Tensor, frames_padding_mask:torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of frames
        returns: Tensor of embeddings for each token (128 tokens of frames)
        """

        def transformer_checkpoint(x):
            return self.transformer(x, src_key_padding_mask=frames_padding_mask)

        B, T, N, C = x.shape                # x -> [batch_size, T, input_size]
        # print(f"Input shape: {x.shape}, Frames padding mask shape: {frames_padding_mask.shape}")
        x = x.permute(0, 3, 1, 2)           # [B, C, T, N]
        assert not torch.isnan(x).any(), "NaNs justo al iniciar"

        # print(f"Permuted input shape: {x.shape}")
        x = self.stgcn(x)                   # [B, 3*out_channels, T, K]
        # print(f"ST-GCN output shape: {x.shape}")
        assert not torch.isnan(x).any(), "NaNs después del ST-GCN"

        x = self.linear_hidden(x)           # [B, hidden, T, K]
        # print(f"Linear hidden output shape: {x.shape}")
        assert not torch.isnan(x).any(), "NaNs después de linear_hidden"

        x = x.mean(dim=-1)                  # [B, hidden, T]
        x = x.permute(0, 2, 1).contiguous() # [B, T, hidden]
        # print(f"Permuted linear hidden output shape: {x.shape}")
        assert not torch.isnan(x).any(), "NaNs después del pool nodos"

        if self.training:
            x = checkpoint(transformer_checkpoint, x, use_reentrant=False)
        else:
            x = transformer_checkpoint(x)  # [B, pool_dim, hidden]

        assert not torch.isnan(x).any(), "NaNs después del transformer"

        Q = self.token_queries.unsqueeze(0).expand(B, -1, -1)   # [B, n_tokens, output_size]

        attn_out, attn_w = self.cross_attn(
            query=Q,
            key=x,
            value=x,
            key_padding_mask=frames_padding_mask
        )  # [B, n_tokens, hidden]
        # print(f"Cross attention output shape: {attn_out.shape}, Attention weights shape: {attn_w.shape}")
        x = self.norm_attn(Q + attn_out)
        x = self.proj(x)     # [B, n_tokens, output_size]
        # print(f"Final output shape: {x.shape}")
        return x, attn_w


class KeypointEncoder(nn.Module):
    """
    Codificador que procesa la secuencia de keypoints.
    Utiliza bloques STGCN para extraer características espacio-temporales
    y un LSTM para modelar la secuencia final.
    """
    def __init__(self, num_nodes, in_coords, stgcn_channels, lstm_hidden_dim, num_stgcn_layers=3):
        super().__init__()

        self.num_nodes = num_nodes

        # Capa inicial para proyectar las coordenadas a una dimensión mayor
        self.input_proj = nn.Conv2d(in_coords, stgcn_channels[0], 1)

        self.stgcn_layers = nn.ModuleList()
        for i in range(num_stgcn_layers):
            in_c = stgcn_channels[i]
            out_c = stgcn_channels[i+1]
            self.stgcn_layers.append(STGCNBlock(in_c, out_c, kernel_size_t=9))

        # LSTM para capturar la dinámica secuencial final
        self.lstm = nn.LSTM(
            input_size=num_nodes * stgcn_channels[-1],
            hidden_size=lstm_hidden_dim,
            batch_first=True,
            num_layers=2,
            bidirectional=True # Bidireccional para capturar contexto de toda la secuencia
        )

        # Proyección final para que la salida del LSTM coincida con la del decoder
        self.output_proj = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim)


    def forward(self, x, A):
        """
        Forward pass.
        - x: Tensor de keypoints [Batch, Frames, Nodos, Coordenadas]
        - A: Matriz de adyacencia [Nodos, Nodos]
        """
        # Ajustar la forma de entrada para Conv2d: [B, C, F, N]
        x = x.permute(0, 3, 1, 2)
        x = self.input_proj(x)

        for layer in self.stgcn_layers:
            x = layer(x, A)

        # Preparar para el LSTM: [B, F, N*C]
        b, c, f, n = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, f, c * n)

        # Pasa por el LSTM
        # La salida del LSTM es (output, (hidden, cell))
        # output tiene forma [Batch, Frames, Hidden_dim * 2]
        output, _ = self.lstm(x)

        # Proyectar la salida a la dimensión deseada
        output = self.output_proj(output)

        return output

# --- Componente 3: Decodificador de Alineamiento ---
# Utiliza el TransformerDecoder de PyTorch para generar la secuencia de embeddings.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AlignmentDecoder(nn.Module):
    """
    Decodificador Transformer que genera la secuencia de embeddings alineados.
    """
    def __init__(self, embed_dim, nhead, num_decoder_layers, dim_feedforward, memory_dim):
        super().__init__()

        # Proyectar la memoria del codificador a la dimensión del decodificador
        self.memory_proj = nn.Linear(memory_dim, embed_dim)

        self.pos_encoder = PositionalEncoding(embed_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True # ¡Importante!
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # El cabezal de salida ya está implícito, la salida del decoder es la predicción
        # No se necesita capa lineal final si memory_dim == embed_dim

    def forward(self, tgt, memory, tgt_mask):
        """
        Forward pass.
        - tgt: Embeddings objetivo (desplazados) [Batch, Tokens_len, Embed_dim]
        - memory: Salida del codificador de keypoints [Batch, Frames, Memory_dim]
        - tgt_mask: Máscara para la atención causal [Tokens_len, Tokens_len]
        """
        # Proyectar la memoria a la dimensión del embedding y aplicar positional encoding a la entrada
        memory = self.memory_proj(memory)
        # La entrada al decoder (tgt) debe tener positional encoding
        # La forma debe ser [Seq, Batch, Dim] para PositionalEncoding
        tgt = self.pos_encoder(tgt.permute(1,0,2)).permute(1,0,2)

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        return output

# --- Modelo Principal ---
class SignAlignGCN_KeypointsOnly(nn.Module):
    def __init__(self, keypoint_encoder, alignment_decoder, adjacency_matrix, embed_dim, bos_token_id=0):
        super().__init__()
        self.encoder = keypoint_encoder
        self.decoder = alignment_decoder
        self.A = adjacency_matrix

        # Embedding para el token de inicio de secuencia (<bos>)
        self.bos_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))

    def _create_target_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, keypoints, target_embeddings=None, max_len=50):
        """
        - keypoints: [B, F, N, C]
        - A: [N, N]
        - target_embeddings: [B, T_len, E_dim] (usado para teacher forcing)
        - max_len: Longitud máxima de la secuencia a generar en inferencia
        """
        # 1. Codificar los keypoints para obtener el contexto del video
        memory = self.encoder(keypoints, self.A)

        if self.training and target_embeddings is not None:
            # --- MODO ENTRENAMIENTO (TEACHER FORCING) ---
            # Preparamos la entrada del decodificador: <bos> + secuencia objetivo
            bos = self.bos_embedding.expand(target_embeddings.size(0), -1, -1)
            decoder_input = torch.cat([bos, target_embeddings[:, :-1, :]], dim=1)

            # Creamos la máscara causal para el decodificador
            tgt_mask = self._create_target_mask(decoder_input.size(1)).to(keypoints.device)

            # Obtenemos la predicción del decodificador
            predicted_embeddings = self.decoder(decoder_input, memory, tgt_mask)
            return predicted_embeddings
        else:
            # --- MODO INFERENCIA (AUTORREGRESIVO) ---
            batch_size = keypoints.size(0)

            # Empezamos con el token <bos>
            decoder_input = self.bos_embedding.expand(batch_size, -1, -1)

            # Almacenamos las salidas
            output_embeddings = []

            for _ in range(max_len):
                tgt_mask = self._create_target_mask(decoder_input.size(1)).to(keypoints.device)

                # Predecir el siguiente embedding
                prediction = self.decoder(decoder_input, memory, tgt_mask)

                # Solo nos interesa el último embedding de la secuencia predicha
                next_embedding = prediction[:, -1:, :]
                output_embeddings.append(next_embedding)

                # Añadir la predicción a la entrada para el siguiente paso
                decoder_input = torch.cat([decoder_input, next_embedding], dim=1)

            return torch.cat(output_embeddings, dim=1)


class PrefixImitator(nn.Module):
    """Wraps Imitator + PrefixAdapter for v116 soft-prefix training.

    forward(keypoints, frames_padding_mask) -> prefix [B, K, hidden_size]

    Scale note: GemmaScaledWordEmbedding returns token embeddings with L2 norm
    ≈ sqrt(hidden_size) ≈ 45.25 for E2B.  The adapter output is normalized to
    unit norm and then rescaled by the learnable ``prefix_scale`` parameter,
    which is initialized to sqrt(hidden_size) so the prefix starts in the
    correct embedding scale without manual tuning.
    """
    def __init__(self, imitator: Imitator, hidden_size: int | None = None):
        super().__init__()
        self.imitator = imitator
        if hidden_size is None:
            hidden_size = imitator.proj.out_features
        self.prefix_adapter = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        # Bug-2 fix: initialize scale to sqrt(hidden_size) so prefix lives in
        # the same L2-norm range as Gemma token embeddings (~45.25 for E2B).
        self.prefix_scale = nn.Parameter(torch.tensor(float(hidden_size) ** 0.5))

    def forward(self, keypoints, frames_padding_mask):
        # raw_prefix: [B, K, hidden_size] from Imitator
        raw_prefix, attn_w = self.imitator(keypoints, frames_padding_mask)
        # Project into Gemma's embedding space with correct norm.
        # F.normalize gives unit-norm vectors; prefix_scale brings them to the
        # same magnitude as token embeddings (≈ sqrt(hidden_size), learnable).
        prefix = self.prefix_adapter(raw_prefix)
        prefix = F.normalize(prefix, dim=-1) * self.prefix_scale
        return prefix, attn_w
