"""Split persistente de dataset2 agrupado por `source_group` (v125).

No usa `random_split` por clip: agrupa clips por `source_group` (programa o
video original) y asigna GRUPOS COMPLETOS a train/val/test con un bin-packing
greedy determinista, de forma que ningún grupo quede repartido entre
particiones. El reparto se hace por número de CLIPS (no de grupos) para que
los ratios objetivo se respeten a nivel de dataset.
"""
import random
from collections import defaultdict


def make_grouped_split(
    clip_ids: list[str],
    source_groups: list[str],
    seed: int = 23,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[str]]:
    assert len(clip_ids) == len(source_groups)
    assert abs(sum(ratios) - 1.0) < 1e-6

    clips_by_group: dict[str, list[str]] = defaultdict(list)
    for cid, g in zip(clip_ids, source_groups):
        clips_by_group[g].append(cid)

    groups = list(clips_by_group.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)
    # grupos grandes primero: el bin-packing greedy se acerca más a los ratios
    # objetivo cuando los ítems grandes se colocan antes que el remanente chico.
    groups.sort(key=lambda g: len(clips_by_group[g]), reverse=True)

    n_total = len(clip_ids)
    targets = {"train": ratios[0] * n_total, "val": ratios[1] * n_total, "test": ratios[2] * n_total}
    counts = {"train": 0, "val": 0, "test": 0}
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for g in groups:
        # asigna el grupo a la partición con mayor déficit relativo respecto a su target
        name = max(("train", "val", "test"), key=lambda n: targets[n] - counts[n])
        split[name].extend(clips_by_group[g])
        counts[name] += len(clips_by_group[g])

    return split
