"""
Unit tests for tournament/blade_rack.py (Phase 3).

All tests are mock-based — no model downloads required.

Tests:
  1. BladeRack.swap() returns ReconfigurationProfile with correct fields
  2. Swap time is measured (non-negative)
  3. Swapping to non-loaded blade raises KeyError
  4. adapter_params is a non-negative integer
  5. MoDStyleRetrainEstimate computes sensible values
  6. Multiple swaps work correctly (cycling through blades)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import torch
from unittest.mock import MagicMock

from tournament.blade_rack import (
    BladeRack,
    ReconfigurationProfile,
    MoDStyleRetrainEstimate,
    _count_lora_params,
    _get_memory_mb,
)


BLADE_NAMES = ["helpfulness", "harmlessness", "truthfulness"]


def _make_mock_rack():
    """Create a BladeRack with mocked blades (no actual downloads)."""
    mock_cfg = MagicMock()
    mock_cfg.blade_sources = {b: {} for b in BLADE_NAMES}
    mock_cfg.beta = 0.1

    mock_tok = MagicMock()
    mock_base = MagicMock()

    rack = BladeRack.__new__(BladeRack)
    rack.cfg = mock_cfg
    rack.tokenizer = mock_tok
    rack.base_model = mock_base
    rack._blades = {}
    rack._active_blade_name = None
    rack._load_times = {}

    # Inject mock DPOBlades directly (bypass load_blade_model)
    from tournament.blades import DPOBlade
    for name in BLADE_NAMES:
        mock_blade_inst = MagicMock(spec=DPOBlade)
        # Make blade_model have named_parameters that include 'lora' keys
        mock_lora_param = torch.nn.Parameter(torch.randn(100, 64))
        mock_blade_inst.blade_model = MagicMock()
        mock_blade_inst.blade_model.named_parameters = lambda: [
            ("lora_A.weight", mock_lora_param),
            ("lora_B.weight", mock_lora_param),
        ]
        rack._blades[name] = mock_blade_inst

    return rack


def test_swap_returns_profile():
    """swap() must return (DPOBlade, ReconfigurationProfile)."""
    rack = _make_mock_rack()
    blade, profile = rack.swap("helpfulness")
    assert isinstance(profile, ReconfigurationProfile), (
        f"Expected ReconfigurationProfile, got {type(profile)}"
    )
    print(f"  ✓ swap() returns ReconfigurationProfile")


def test_profile_fields():
    """ReconfigurationProfile must have all required fields."""
    rack = _make_mock_rack()
    _, profile = rack.swap("harmlessness")

    assert profile.to_blade == "harmlessness"
    assert profile.from_blade == "<none>"
    assert profile.swap_time_ms >= 0.0
    assert isinstance(profile.adapter_params, int)
    assert profile.adapter_params >= 0
    print(f"  to_blade: {profile.to_blade}  ✓")
    print(f"  swap_time_ms: {profile.swap_time_ms:.4f}  ✓")
    print(f"  adapter_params: {profile.adapter_params:,}  ✓")


def test_swap_time_measured():
    """Swap time should be non-negative and very small (pointer swap)."""
    rack = _make_mock_rack()
    _, profile = rack.swap("helpfulness")
    assert profile.swap_time_ms >= 0.0, f"Negative swap time: {profile.swap_time_ms}"
    # Pointer swap should be << 1ms
    assert profile.swap_time_ms < 100.0, (
        f"Swap time suspiciously large: {profile.swap_time_ms} ms"
    )
    print(f"  swap_time_ms = {profile.swap_time_ms:.4f}ms  (< 100ms)  ✓")


def test_swap_unloaded_blade_raises():
    """Swapping to a blade that isn't loaded should raise KeyError."""
    rack = _make_mock_rack()
    rack._blades = {}  # empty the rack

    try:
        rack.swap("helpfulness")
        assert False, "Should have raised KeyError"
    except KeyError as e:
        print(f"  ✓ Correctly raised KeyError: {e}")


def test_active_blade_name_updated():
    """After swap(), active_blade_name must reflect the new blade."""
    rack = _make_mock_rack()
    assert rack.active_blade_name is None

    rack.swap("helpfulness")
    assert rack.active_blade_name == "helpfulness"

    rack.swap("harmlessness")
    assert rack.active_blade_name == "harmlessness"

    rack.swap("truthfulness")
    assert rack.active_blade_name == "truthfulness"
    print("  ✓ active_blade_name tracks swaps correctly")


def test_cycling_swaps():
    """Swapping through all blades multiple times should work without error."""
    rack = _make_mock_rack()
    N_rounds = 9
    for i in range(N_rounds):
        target = BLADE_NAMES[i % len(BLADE_NAMES)]
        _, profile = rack.swap(target)
        assert profile.to_blade == target
    print(f"  ✓ {N_rounds} swap cycles completed without error")


def test_mod_retrain_estimate_sensible():
    """MoDStyleRetrainEstimate must produce sensible values."""
    est = MoDStyleRetrainEstimate()
    summary = est.summary()

    assert est.router_params > 0, "Router params must be positive"
    assert est.joint_pathway_params > 0, "Joint pathway params must be positive"
    assert est.gpu_hours_estimate > 0, "GPU-hours must be positive"
    assert "note" in summary, "Summary must include explanation note"

    print(f"  router_params:        {summary['router_params']}")
    print(f"  joint_pathway_params: {summary['joint_pathway_params']}")
    print(f"  gpu_hours_estimate:   {summary['gpu_hours_estimate']}")
    print("  ✓ MoDStyleRetrainEstimate values are sensible")


def test_count_lora_params():
    """_count_lora_params should count parameters with 'lora' in name."""
    mock_model = MagicMock()
    lora_a = torch.nn.Parameter(torch.randn(100, 64))
    lora_b = torch.nn.Parameter(torch.randn(64, 100))
    other  = torch.nn.Parameter(torch.randn(512, 512))

    mock_model.named_parameters = lambda: [
        ("lora_A.weight", lora_a),
        ("lora_B.weight", lora_b),
        ("mlp.weight", other),  # not LoRA
    ]

    count = _count_lora_params(mock_model)
    expected = lora_a.numel() + lora_b.numel()
    assert count == expected, f"Expected {expected}, got {count}"
    print(f"  ✓ LoRA param count = {count:,} (non-LoRA params excluded)")


def test_profile_str():
    """ReconfigurationProfile.__str__ should be human-readable."""
    profile = ReconfigurationProfile(
        from_blade="helpfulness",
        to_blade="harmlessness",
        swap_time_ms=0.05,
        memory_before_mb=1200.0,
        memory_after_mb=1200.1,
        memory_delta_mb=0.1,
        adapter_params=7_340_032,
    )
    s = str(profile)
    assert "harmlessness" in s
    assert "helpfulness" in s
    print(f"  ✓ Profile str: {s}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Swiss Knife — BladeRack Unit Tests")
    print("=" * 60)
    print()

    print("TEST 1 — swap() returns profile")
    test_swap_returns_profile()
    print()

    print("TEST 2 — Profile fields")
    test_profile_fields()
    print()

    print("TEST 3 — Swap time measured")
    test_swap_time_measured()
    print()

    print("TEST 4 — Unloaded blade raises KeyError")
    test_swap_unloaded_blade_raises()
    print()

    print("TEST 5 — active_blade_name updated")
    test_active_blade_name_updated()
    print()

    print("TEST 6 — Cycling swaps")
    test_cycling_swaps()
    print()

    print("TEST 7 — MoD retrain estimate sensible")
    test_mod_retrain_estimate_sensible()
    print()

    print("TEST 8 — _count_lora_params")
    test_count_lora_params()
    print()

    print("TEST 9 — Profile str format")
    test_profile_str()
    print()

    print("=" * 60)
    print("  ALL BLADE RACK TESTS PASSED ✓")
    print("=" * 60)
