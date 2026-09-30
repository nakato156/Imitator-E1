from torch.utils.data import Sampler
from transformers.trainer_pt_utils import get_length_grouped_indices

class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size: int, lengths: list = None, mega_batch_mult: int = None, generator=None):
        # `lengths` debería ser la duración temporal real de cada muestra (en
        # frames), no `len(item)` sobre la tupla (keypoint, embedding, label)
        # que siempre da 3 -- ver KeypointDataset.split_dataset, que ya la
        # calcula correctamente (incluye el reescalado de las augmentations).
        if lengths is None:
            lengths = [len(item) for item in dataset]
        self.indices = get_length_grouped_indices(
            lengths=lengths,
            batch_size=batch_size,
            mega_batch_mult=mega_batch_mult,
            generator=generator,
        )

        self.batches = [
            self.indices[i : i + batch_size]
            for i in range(0, len(self.indices), batch_size)
        ]

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)