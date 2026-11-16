from __future__ import annotations

import json
from pathlib import Path

from rl_optimization.experiments.batch import run_batch_comparison


def test_batch_run_creates_summary_and_run_files(tmp_path: Path) -> None:
    out = run_batch_comparison(
        config_path="experiments/configs/rl_de_comparison_check.yaml",
        repeats=1,
        output_dir=tmp_path,
        debug=False,
        algorithms=["de_rand_1", "rl_de_q_learning"],
        functions=["sphere", "ackley"],
    )

    summary_json = Path(out["summary_json"])
    summary_csv = Path(out["summary_csv"])
    assert summary_json.exists()
    assert summary_csv.exists()

    data = json.loads(summary_json.read_text(encoding="utf-8"))
    assert data["repeats_used"] == 1
    assert set(data["functions"]) == {"sphere", "ackley"}
    assert set(data["algorithms"]) == {"de_rand_1", "rl_de_q_learning"}
    assert "mean_rank" in data
    assert "rl_stage_frequencies_ratio" in data
    assert set(data["rl_stage_frequencies_ratio"].keys()) == {"early", "middle", "late"}
    for state in ("early", "middle", "late"):
        assert set(data["rl_stage_frequencies_ratio"][state].keys()) == {
            "rand_1",
            "best_1",
            "current_to_best_1",
        }

    run_root = Path(data["run_files_root"])
    assert run_root.exists()
    run_files = list(run_root.rglob("run_*.json"))
    assert len(run_files) == 4


def test_batch_run_uses_independent_runs_from_config_when_repeats_none(tmp_path: Path) -> None:
    out = run_batch_comparison(
        config_path="experiments/configs/rl_de_comparison_check.yaml",
        repeats=None,
        output_dir=tmp_path,
        debug=False,
        algorithms=["de_rand_1"],
        functions=["sphere"],
    )
    data = json.loads(Path(out["summary_json"]).read_text(encoding="utf-8"))
    assert data["repeats_used"] == 1
    run_files = list(Path(data["run_files_root"]).rglob("run_*.json"))
    assert len(run_files) == 1


def test_batch_run_resume_skips_completed_runs(tmp_path: Path) -> None:
    out1 = run_batch_comparison(
        config_path="experiments/configs/rl_de_comparison_check.yaml",
        repeats=1,
        output_dir=tmp_path,
        batch_name="resume_case",
        resume=False,
        debug=False,
        algorithms=["de_rand_1"],
        functions=["sphere"],
    )
    data1 = json.loads(Path(out1["summary_json"]).read_text(encoding="utf-8"))
    assert data1["executed_runs"] == 1
    assert data1["skipped_completed_runs"] == 0

    out2 = run_batch_comparison(
        config_path="experiments/configs/rl_de_comparison_check.yaml",
        repeats=1,
        output_dir=tmp_path,
        batch_name="resume_case",
        resume=True,
        debug=False,
        algorithms=["de_rand_1"],
        functions=["sphere"],
    )
    data2 = json.loads(Path(out2["summary_json"]).read_text(encoding="utf-8"))
    assert data2["executed_runs"] == 0
    assert data2["skipped_completed_runs"] == 1
