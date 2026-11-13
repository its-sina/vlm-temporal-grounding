import argparse
import json

import pytest

from src.data.annotations import dev_subset, find_video, load_samples, select_samples
from src.utils import paths
from src.utils.config import parse_args_with_config
from src.utils.io import load_predictions, parse_shard, read_jsonl_tolerant


@pytest.fixture
def annotation_file(tmp_path):
    data = {"vidA": {"duration": 30.0, "timestamps": [[1, 5], [10, 20]], "sentences": ["a person sits.", "b"]},
            "vidB": {"duration": 12.5, "timestamps": [[0, 3]], "sentences": ["c."]}}
    p = tmp_path / "test.json"
    p.write_text(json.dumps(data))
    return p


def test_load_samples_and_subsets(annotation_file):
    s = load_samples(annotation_file)
    assert [x.uid for x in s] == ["00000_vidA", "00001_vidA", "00002_vidB"]
    assert s[0].query == "a person sits" and s[0].gt == (1.0, 5.0)
    assert dev_subset(s, 2) == dev_subset(s, 2)
    assert [x.idx for x in dev_subset(s, 2)] == sorted(x.idx for x in dev_subset(s, 2))
    assert len(select_samples(s, shard=(1, 2))) == 1


def test_find_video(tmp_path):
    (tmp_path / "abc.mkv").write_bytes(b"")
    assert find_video(tmp_path, "abc") == tmp_path / "abc.mkv"
    assert find_video(tmp_path, "nope") is None


def test_config_defaults_and_cli_override(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("stride: 12\nmethod: iso+dedup\nmax_new_tokens: 64\n")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=7)
    ap.add_argument("--method", default="full")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    a = parse_args_with_config(ap, ["--config", str(cfg), "--stride", "6"])
    assert (a.stride, a.method, a.max_new_tokens) == (6, "iso+dedup", 64)


def test_config_rejects_unknown_keys(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("strid: 12\n")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=7)
    with pytest.raises(SystemExit):
        parse_args_with_config(ap, ["--config", str(cfg)])


def test_shipped_configs_are_valid():
    from experiments.run_grounding import build_parser
    for cfg in sorted((paths.PROJECT_ROOT / "configs").rglob("*.yaml")):
        a = parse_args_with_config(build_parser(), ["--config", str(cfg), "--exp", "x"])
        assert a.method.partition("+")[0] in ("seq", "full", "iso", "hybrid", "frameseq", "bbridge"), cfg


def test_env_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("VTG_CACHE_DIR", str(tmp_path))
    assert paths.frame_cache_dir("charades", 1.0, 336) == tmp_path / "frames" / "charades_f1.0_ts336"
    monkeypatch.delenv("VTG_DATA_DIR", raising=False)
    assert paths.default_video_dir("anet") == paths.PROJECT_ROOT / "data" / "anet" / "videos"


def test_tolerant_jsonl_and_predictions(tmp_path):
    p = tmp_path / "pred_shard0.jsonl"
    p.write_text('{"uid": "a", "iou": 0.5}\n{"uid": "b", "iou"')
    assert [r["uid"] for r in read_jsonl_tolerant(p)] == ["a"]
    assert p.read_text() == '{"uid": "a", "iou": 0.5}\n'
    assert set(load_predictions(tmp_path)) == {"a"}
    assert parse_shard("1/3") == (1, 3)
    with pytest.raises(ValueError):
        parse_shard("3/3")
