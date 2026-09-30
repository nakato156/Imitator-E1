import h5py
import numpy as np
import torch
from typing import Optional, List, Tuple
from torch.utils.data import random_split, Dataset, Subset, ConcatDataset
from .data_augmentation import normalize_augment_data, remove_keypoints, TEMPORAL_DROP_PROB
from .vocab import tokenize
class TransformedSubset(Dataset):
    def __init__(self, subset: Subset, transform_fn: str, return_label=False, video_lengths=[], n_keypoints=133):
        self.subset    = subset
        self.transform = transform_fn
        self.return_label = return_label
        self.video_lengths = video_lengths
        self.n_keypoints = n_keypoints

        if self.transform == "Length_variance":
            self.video_lengths = [int(round(0.8 * video)) for video in self.video_lengths]
        elif self.transform == "Temporal_drop":
            self.video_lengths = [int(round((1 - TEMPORAL_DROP_PROB) * video)) for video in self.video_lengths]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        keypoint, embedding, label = self.subset[idx]

        keypoint = normalize_augment_data(keypoint, self.transform, self.n_keypoints)

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)

        if self.return_label:
            return keypoint, embedding, label

        return keypoint, embedding, None

class KeypointDataset(Dataset):
    # Datasets usados por defecto (signos aislados/glosas). Para el experimento SIGReg
    # se pasa include_datasets=["dataset2"] (frases, LSA) como dataset principal.
    DEFAULT_INCLUDE_DATASETS = ["dataset1", "dataset3", "dataset5", "dataset7"]

    def __init__(self, h5Path, n_keypoints=111, return_label=False, max_length=4000, data_augmentation=True, include_datasets=None, return_token_ids=False, text_group="embeddings", min_frames=0, filter_invalid_labels=False):
        self.h5Path = h5Path
        self.n_keypoints = n_keypoints
        self.return_label = return_label
        # Filtro de calidad de datos (v119, "Less is More" generalizado a la
        # restricción real de nn.CTCLoss en esta arquitectura). Default no-op:
        # min_frames=0 y filter_invalid_labels=False no cambian valid_index
        # respecto al comportamiento previo (protege a otros call-sites, p.ej.
        # setup_train.py, que no pasan estos kwargs).
        self.min_frames = min_frames
        self.filter_invalid_labels = filter_invalid_labels
        # Experimento CE-vocab: devuelve los token IDs (grupo `token_ids` del HDF5,
        # generado por scripts/data/add_token_ids_h5.py) en el tercer slot de la tupla.
        self.return_token_ids = return_token_ids
        # Grupo HDF5 de donde leer el "embedding" de texto en __getitem__. Por defecto
        # "embeddings" (tabla de entrada de Gemma, per-token, usado por CE-AR v116/v117).
        # v118e usa "text_ctx" (último hidden state contextual, 1 vector por clip) porque
        # el mean-pool de input-embeddings resultó tener techo de retrieval bajo: ver
        # memoria v118-text-discriminability.
        self.text_group = text_group
        self.max_length = max_length
        self.video_lengths = []
        self.data_augmentation = data_augmentation
        self.include_datasets = include_datasets if include_datasets else self.DEFAULT_INCLUDE_DATASETS
    
        self.data_augmentation_dict = {
            0: "Length_variance",
            1: "Gaussian_jitter",
            2: "Rotation_2D",
            4: "Scaling",
            5: "Temporal_drop",
        }

        self.dataset_length = 0
        self.processData()

    def processData(self):
        with h5py.File(self.h5Path, 'r') as f:
            datasets  = list(f.keys())
            datasets = sorted(datasets)
        
            self.valid_index = []
            self.original_videos = []

            for dataset in datasets:
                if dataset not in self.include_datasets:
                    continue

                clip_ids  = list(f[dataset]["embeddings"].keys())

                for clip in clip_ids:
                    try:
                        if self.return_token_ids and clip not in f[dataset]["token_ids"]:
                            continue
                        shape = f[dataset]["keypoints"][clip].shape[0]
                        if shape < self.max_length and shape >= self.min_frames:
                            if self.filter_invalid_labels:
                                label = f[dataset]["labels"][clip][:][0].decode()
                                n_tokens = len(tokenize(label))
                                if n_tokens == 0 or shape // 4 < n_tokens:
                                    continue
                            self.valid_index.append((dataset, clip))
                            self.video_lengths.append(shape)
                    except KeyError:
                        print(f"KeyError for {dataset}/{clip}, skipping...")
                        continue
                
            self.dataset_length = len(self.valid_index)

    def split_dataset(self, train_ratio):
        train_dataset, validation_dataset = random_split(self, [train_ratio, 1 - train_ratio], generator=torch.Generator().manual_seed(42))
        val_length = [self.video_lengths[i] for i in validation_dataset.indices] 
        
        if self.data_augmentation:
            train_length = [self.video_lengths[i] 
                            for i in train_dataset.indices]
            train_subset = Subset(self, train_dataset.indices)
            aug_subsets = [
                TransformedSubset(train_subset,
                                  transform_fn=tf,
                                  return_label=self.return_label,
                                  video_lengths=train_length,
                                  n_keypoints=self.n_keypoints
                                  )
                for tf in self.data_augmentation_dict.values()
            ]

            trains_subset_length = [ length
                for subset in aug_subsets
                for length in subset.video_lengths
            ]
            
            train_lengths = train_length + trains_subset_length 
            train_dataset = ConcatDataset([train_subset, *aug_subsets])
            
            self.dataset_length = len(val_length) + len(train_length)
        else:
            train_lengths = [self.video_lengths[i] for i in train_dataset.indices]

        print("Videos: ", self.dataset_length)
        return train_dataset, validation_dataset, train_lengths, val_length

    def __len__(self):
        return len(self.valid_index)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Optional[np.ndarray], torch.Tensor, Optional[str]]:
        """
        Recupera una muestra individual del conjunto de datos.
        Este método recupera los puntos clave, la matriz de adyacencia, los embeddings y opcionalmente las etiquetas
        del archivo HDF5 para el índice dado. Los puntos clave se procesan eliminando
        puntos específicos y normalizando/aumentando los datos.
        Args:
            idx (int): Índice de la muestra a recuperar.
        Returns:
            Tupla que contiene:
            - keypoint (torch.Tensor): Datos de puntos clave procesados.
            - A (Optional[np.ndarray]): Matriz de adyacencia que representa el grafo esquelético.
            - embedding (torch.Tensor): Vector de embedding para la muestra.
            - label (Optional[str]): Cadena de etiqueta si return_label es True, None en caso contrario.
        """
        
        mapped_idx = self.valid_index[idx]

        with h5py.File(self.h5Path, 'r') as f:
            keypoint = f[mapped_idx[0]]["keypoints"][mapped_idx[1]][:]
            embedding = f[mapped_idx[0]][self.text_group][mapped_idx[1]][:]

            if self.return_token_ids:
                token_ids = torch.as_tensor(f[mapped_idx[0]]["token_ids"][mapped_idx[1]][:], dtype=torch.long)

            if self.return_label:
                label = f[mapped_idx[0]]["labels"][mapped_idx[1]][:][0].decode()

        keypoint = remove_keypoints(keypoint)
        keypoint = normalize_augment_data(keypoint, "Original", self.n_keypoints)

        if not isinstance(embedding, torch.Tensor):
            embedding = torch.as_tensor(embedding)

        if self.return_token_ids:
            return keypoint, embedding, token_ids

        if self.return_label:
            return keypoint, embedding, label

        return keypoint, embedding, None
