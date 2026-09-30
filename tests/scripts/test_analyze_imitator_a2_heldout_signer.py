from scripts.diagnostics.analyze_imitator_a2 import parse_args
from scripts.train.train_temporal_v126 import split_records


def test_parse_args_defaults_heldout_signer_to_none(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["analyze_imitator_a2.py", "--checkpoint", "x.pt", "--output", "out.json"],
    )
    args = parse_args()
    assert args.heldout_signer is None


def test_parse_args_accepts_heldout_signer(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_imitator_a2.py",
            "--checkpoint", "x.pt",
            "--output", "out.json",
            "--heldout-signer", "3",
        ],
    )
    args = parse_args()
    assert args.heldout_signer == 3


def test_split_records_heldout_signer_isolates_one_signer():
    records = [{"clip_id": i, "label": "gloss", "signer_id": i % 4} for i in range(20)]
    train, val = split_records(records, seed=23, heldout_signer=2)
    assert all(r["signer_id"] != 2 for r in train)
    assert all(r["signer_id"] == 2 for r in val)
    assert len(train) + len(val) == len(records)
