"""Environment diagnostic regressions."""

from pathlib import Path

from florence_forge.utils.diagnostics import (
    DependencyCheck,
    collect_environment_diagnostics,
    find_local_hf_snapshot,
)


def test_find_local_hf_snapshot_prefers_refs_main(tmp_path):
    model_dir = tmp_path / "models--microsoft--Florence-2-base"
    revision = "abc123"
    snapshot = model_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    refs = model_dir / "refs"
    refs.mkdir()
    (refs / "main").write_text(revision, encoding="utf-8")

    assert find_local_hf_snapshot("microsoft/Florence-2-base", cache_root=tmp_path) == snapshot


def test_collect_environment_diagnostics_can_require_local_model(tmp_path):
    report = collect_environment_diagnostics(
        requested_device="cpu",
        require_model=True,
        cache_root=tmp_path,
        checks=[
            DependencyCheck(
                module="definitely_missing_florenceforge_dep",
                package="definitely-missing-florenceforge-dep",
                required=True,
            )
        ],
    )

    assert report["ok"] is False
    assert report["missing_required"] == ["definitely-missing-florenceforge-dep"]
    assert report["model"]["local_snapshot_exists"] is False
    assert "Required local model snapshot is missing." in report["warnings"]
