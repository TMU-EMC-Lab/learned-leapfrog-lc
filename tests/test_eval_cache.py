"""評価の最適化（2026-10-21）：参照解とロールアウトを最長の t_eval で 1 回だけ計算して切り出しても，t_eval ごとに
計算したときと同じ指標になること，参照解の保存（float32）を使っても指標がほぼ変わらないこと."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import system_eval as se  # noqa: E402
from harmonic_ic_dataset import DATASETS_DIR  # noqa: E402
from harmonic_models import build_model  # noqa: E402

CASES = [("nlqf3_v1_dtd2", "LF_Free_MLP", "lf", 2), ("nlqf3_v1_dtd2", "Res_MLP_state", "direct_state", 2),
         ("nltl8_v1_dtd2", "LF_Free_Loc", "lf", 16), ("nltl8_v1_dtd2", "Res_Loc_state", "direct_state", 16)]


def _small_manifest(ds, n=6):
    m = json.load(open(Path(DATASETS_DIR) / ds / "manifest.json"))
    for sp in m["splits"].values():
        for k in list(sp):
            sp[k] = sp[k][:n]
    return m


@pytest.mark.parametrize("ds,name,family,dim", CASES)
def test_sliced_equals_separate(ds, name, family, dim, tmp_path, monkeypatch):
    monkeypatch.setenv("BHIC_NO_REF_CACHE", "1")
    torch.manual_seed(0)
    model = build_model(name, 8, dim=dim).to(se.ev.DEVICE)
    m = _small_manifest(ds)
    both = se.evaluate(model, family, m, ["test_interp"], [10.0, 50.0], tmp_path, {})
    only = se.evaluate(model, family, m, ["test_interp"], [10.0], tmp_path, {})
    for k, v in only[0].items():
        if k.endswith(("_mean", "_median", "_max")):
            assert both[0][k] == pytest.approx(v, rel=1e-6, abs=1e-12, nan_ok=True), k


@pytest.mark.parametrize("ds,name,family,dim", CASES[:3])
def test_reference_cache(ds, name, family, dim, tmp_path, monkeypatch):
    torch.manual_seed(1)
    model = build_model(name, 8, dim=dim).to(se.ev.DEVICE)
    m = _small_manifest(ds)
    m["dataset_id"] = "zz_test_ref_cache"
    try:
        monkeypatch.setenv("BHIC_NO_REF_CACHE", "1")
        a = se.evaluate(model, family, m, ["test_interp"], [10.0], tmp_path, {})
        monkeypatch.delenv("BHIC_NO_REF_CACHE")
        b = se.evaluate(model, family, m, ["test_interp"], [10.0], tmp_path, {})  # 保存する
        c = se.evaluate(model, family, m, ["test_interp"], [10.0], tmp_path, {})  # 保存したものを読む
        assert (Path(DATASETS_DIR) / "zz_test_ref_cache" / "ref_cache").exists()
        for k, v in a[0].items():
            if k.endswith("x_rmse_rel_mean"):
                assert b[0][k] == pytest.approx(v, rel=1e-9)
                assert c[0][k] == pytest.approx(v, rel=1e-4)
    finally:
        shutil.rmtree(Path(DATASETS_DIR) / "zz_test_ref_cache", ignore_errors=True)
