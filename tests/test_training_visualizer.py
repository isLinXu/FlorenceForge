"""TrainingVisualizer 冒烟测试。"""

from pathlib import Path

import pytest

from florence_forge.training.visualizer import TrainingVisualizer, VISUALIZATION_AVAILABLE


def test_init_creates_output_dir(tmp_path):
    viz = TrainingVisualizer(str(tmp_path / "viz_out"))
    assert viz.output_dir.exists()


def test_plot_loss_curves_missing_csv_returns_empty(tmp_path):
    viz = TrainingVisualizer(str(tmp_path))
    assert viz.plot_loss_curves() == ""


@pytest.mark.skipif(not VISUALIZATION_AVAILABLE, reason="需要 matplotlib 与 pandas")
def test_plot_loss_curves_with_minimal_csv(tmp_path):
    out = tmp_path / "metrics"
    out.mkdir()
    (out / "epoch_metrics.csv").write_text(
        "epoch,train_loss,val_loss,learning_rate\n0,1.0,0.9,0.001\n",
        encoding="utf-8",
    )
    viz = TrainingVisualizer(str(out))
    path = viz.plot_loss_curves()
    assert path
    assert Path(path).exists()
