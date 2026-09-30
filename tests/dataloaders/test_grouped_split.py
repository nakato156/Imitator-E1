import random

from src.mslm.dataloader.grouped_split import make_grouped_split


def _synthetic(n_groups=40, clips_per_group=10):
    clip_ids, groups = [], []
    for g in range(n_groups):
        for c in range(clips_per_group):
            clip_ids.append(f"g{g}_c{c}")
            groups.append(f"group_{g}")
    return clip_ids, groups


def test_no_group_spans_two_splits():
    clip_ids, groups = _synthetic()
    split = make_grouped_split(clip_ids, groups, seed=23)
    group_of = dict(zip(clip_ids, groups))

    splits_by_group: dict[str, set[str]] = {}
    for split_name, ids in split.items():
        for cid in ids:
            splits_by_group.setdefault(group_of[cid], set()).add(split_name)

    offenders = {g: s for g, s in splits_by_group.items() if len(s) > 1}
    assert offenders == {}


def test_all_clips_assigned_exactly_once():
    clip_ids, groups = _synthetic()
    split = make_grouped_split(clip_ids, groups, seed=23)
    all_assigned = split["train"] + split["val"] + split["test"]
    assert sorted(all_assigned) == sorted(clip_ids)
    assert len(all_assigned) == len(set(all_assigned))


def test_ratios_are_approximately_respected():
    clip_ids, groups = _synthetic(n_groups=100, clips_per_group=10)
    split = make_grouped_split(clip_ids, groups, seed=23, ratios=(0.8, 0.1, 0.1))
    n_total = len(clip_ids)
    assert abs(len(split["train"]) / n_total - 0.8) < 0.05
    assert abs(len(split["val"]) / n_total - 0.1) < 0.05
    assert abs(len(split["test"]) / n_total - 0.1) < 0.05


def test_deterministic_for_same_seed():
    clip_ids, groups = _synthetic()
    a = make_grouped_split(clip_ids, groups, seed=23)
    b = make_grouped_split(clip_ids, groups, seed=23)
    assert a == b


def test_different_seed_can_change_assignment():
    clip_ids, groups = _synthetic()
    a = make_grouped_split(clip_ids, groups, seed=23)
    b = make_grouped_split(clip_ids, groups, seed=999)
    assert a != b


def test_single_group_goes_entirely_to_train_when_ratios_demand_it():
    # un solo grupo no se puede partir: debe ir completo a alguna partición
    clip_ids = [f"c{i}" for i in range(5)]
    groups = ["only_group"] * 5
    split = make_grouped_split(clip_ids, groups, seed=23)
    non_empty = [name for name, ids in split.items() if ids]
    assert len(non_empty) == 1


def test_uneven_group_sizes_still_avoid_leakage():
    rng = random.Random(0)
    clip_ids, groups = [], []
    for g in range(30):
        size = rng.randint(1, 25)
        for c in range(size):
            clip_ids.append(f"g{g}_c{c}")
            groups.append(f"group_{g}")
    split = make_grouped_split(clip_ids, groups, seed=23)
    group_of = dict(zip(clip_ids, groups))
    seen: dict[str, str] = {}
    for split_name, ids in split.items():
        for cid in ids:
            g = group_of[cid]
            assert seen.get(g, split_name) == split_name
            seen[g] = split_name
