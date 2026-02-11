"""Tests for expert health diagnostics in soft-routing setting."""

from pathlib import Path

import torch

from src.utils.expert_health_diag import (
    ExpertHealthDiagConfig,
    ExpertHealthDiagnostics,
    UtilizationConfig,
    compute_expert_utilization,
    compute_gini_coefficient,
)


def _repeat_row(row, n: int = 128) -> torch.Tensor:
    return torch.tensor([row] * n, dtype=torch.float32)


def test_gini_coefficient():
    gini_uniform = compute_gini_coefficient([0.25, 0.25, 0.25, 0.25])
    gini_skewed = compute_gini_coefficient([0.9, 0.05, 0.03, 0.02])
    assert gini_uniform < 0.1
    assert gini_skewed > 0.5


def test_case1_rank_dominance_not_dead_or_monopoly():
    """Case1: mean_weight near-uniform but argmax always expert_0."""
    gate_w = _repeat_row([0.251, 0.249, 0.250, 0.250], n=256)
    expert_names = ["expert_0", "expert_1", "expert_2", "expert_3"]
    util_metrics = compute_expert_utilization(gate_w, expert_names, UtilizationConfig())

    assert util_metrics["dead_experts"] == []
    assert util_metrics["monopoly_experts"] == []
    assert util_metrics["rank_dominant_experts"] == ["expert_0"]
    assert util_metrics["rank_dominant_count"] == 1


def test_case2_true_dead_by_mean_or_p95():
    """Case2: expert_2 is genuinely soft-dead."""
    gate_w = _repeat_row([0.460, 0.279, 0.001, 0.260], n=256)
    expert_names = ["expert_0", "expert_1", "expert_2", "expert_3"]
    util_metrics = compute_expert_utilization(gate_w, expert_names, UtilizationConfig())

    assert util_metrics["dead_experts"] == ["expert_2"]
    assert "expert_2" in util_metrics["dead_experts_by_mean"]
    assert "expert_2" in util_metrics["dead_experts_by_p95"]
    assert util_metrics["expert_weight_p95"]["expert_2"] <= 0.0011


def test_case3_true_monopoly_not_rank_dominant():
    """Case3: expert_1 has both high top1_share and high mean_weight."""
    gate_w = _repeat_row([0.120, 0.680, 0.110, 0.090], n=256)
    expert_names = ["expert_0", "expert_1", "expert_2", "expert_3"]
    util_metrics = compute_expert_utilization(gate_w, expert_names, UtilizationConfig())

    assert util_metrics["monopoly_experts"] == ["expert_1"]
    assert util_metrics["rank_dominant_experts"] == []
    assert util_metrics["dead_experts"] == []


def test_diagnostics_manager_logs_new_fields():
    gate_w = _repeat_row([0.251, 0.249, 0.250, 0.250], n=128)
    expert_names = ["expert_0", "expert_1", "expert_2", "expert_3"]
    tmp_suffix = int(torch.randint(0, 1_000_000, (1,)).item())
    tmpdir = Path(f"runs/_tmp_diag_test_{tmp_suffix}")
    tmpdir.mkdir(parents=True, exist_ok=True)
    log_path = tmpdir / "expert_health_diag.jsonl"

    cfg = ExpertHealthDiagConfig(enabled=True, log_interval=10, log_on_valid=True)
    diag = ExpertHealthDiagnostics(cfg, tmpdir)

    for _ in range(3):
        diag.collect_gate_weights("ctr", gate_w)
        diag.collect_gate_weights("cvr", gate_w)
    diag.set_expert_names("ctr", expert_names)
    diag.set_expert_names("cvr", expert_names)

    metrics = diag.compute_and_log(step=100, epoch=1, phase="train")
    util_ctr = metrics["utilization_ctr"]
    assert "expert_weight_p95" in util_ctr
    assert "rank_dominant_experts" in util_ctr
    assert metrics["alerts"]["has_alert"] is True
    assert any("rank-dominant" in item for item in metrics["alerts"]["summary"])

    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8").strip()


def test_config_from_dict_and_legacy_compat():
    cfg_dict_new = {
        "enabled": True,
        "log_interval": 500,
        "log_on_valid": True,
        "utilization": {
            "enabled": True,
            "dead_mean_threshold": 0.02,
            "dead_p95_threshold": 0.02,
            "monopoly_top1_threshold": 0.97,
            "monopoly_mean_threshold": 0.6,
            "rank_dom_top1_threshold": 0.95,
            "rank_dom_uniform_eps": 0.03,
            "rank_dom_std_eps": 0.02,
        },
        "output_stats": {"enabled": False},
    }
    cfg_new = ExpertHealthDiagConfig.from_dict(cfg_dict_new)
    assert cfg_new.enabled is True
    assert cfg_new.log_interval == 500
    assert cfg_new.utilization.dead_mean_threshold == 0.02
    assert cfg_new.utilization.monopoly_top1_threshold == 0.97
    assert cfg_new.utilization.monopoly_mean_threshold == 0.6
    assert cfg_new.output_stats.enabled is False

    cfg_dict_legacy = {
        "enabled": True,
        "utilization": {
            "dead_threshold": 0.03,
            "monopoly_threshold": 0.88,
        },
    }
    cfg_legacy = ExpertHealthDiagConfig.from_dict(cfg_dict_legacy)
    assert cfg_legacy.utilization.dead_mean_threshold == 0.03
    assert cfg_legacy.utilization.monopoly_top1_threshold == 0.88


def test_alert_summary_strings_for_soft_dead_and_monopoly():
    names = ["expert_0", "expert_1", "expert_2", "expert_3"]
    dead_case = compute_expert_utilization(
        _repeat_row([0.460, 0.279, 0.001, 0.260], n=128),
        names,
        UtilizationConfig(),
    )
    mono_case = compute_expert_utilization(
        _repeat_row([0.120, 0.680, 0.110, 0.090], n=128),
        names,
        UtilizationConfig(),
    )

    diag = ExpertHealthDiagnostics(ExpertHealthDiagConfig(enabled=False), Path("runs/_tmp_diag_test"))
    alerts = diag._generate_alerts({"util_dead": dead_case, "util_mono": mono_case})
    summary_text = " | ".join(alerts["summary"])

    assert alerts["has_alert"] is True
    assert "soft-dead by mean_weight/p95" in summary_text
    assert "by top1_share+mean_weight" in summary_text
