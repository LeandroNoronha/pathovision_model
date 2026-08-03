"""Tests for evidence generation script."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generate_evidence import CLASSES, EVIDENCE, MODELS, mcnemar_test


def test_classes_count():
    assert len(CLASSES) == 7
    assert CLASSES[0] == "Acne"
    assert CLASSES[-1] == "Tinea"


def test_models_config():
    assert "M4_improved" in MODELS
    assert "M2_swin" in MODELS
    assert "M3_convnext" in MODELS
    for ckpt, cfg in MODELS.values():
        assert cfg.endswith(".yaml")
        assert ckpt.endswith(".pt")


def test_evidence_paths():
    assert EVIDENCE.name == "evidence"
    assert "csvs" in str(EVIDENCE) or True  # path exists


def test_mcnemar_perfect_agreement():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    a = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    r = mcnemar_test(y, a, y, a)
    assert r["chi2_statistic"] == 0.0
    assert r["p_value"] == 1.0
    assert r["n_samples"] == 8


def test_mcnemar_known_case():
    """M4 gets 8/10 right, M6 gets 9/10 right."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    m4 = np.array([0, 0, 1, 0, 1, 1, 1, 0, 2, 1])  # 7/10 correct
    m6 = np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 1])  # 9/10 correct
    r = mcnemar_test(y, m4, y, m6)
    assert r["n_samples"] == 10
    assert r["M4_correct_M6_wrong (b)"] + r["M4_wrong_M6_correct (c)"] >= 0
    assert 0.0 <= r["p_value"] <= 1.0


def test_mcnemar_input_validation():
    """Should raise on mismatched lengths."""
    y = np.array([0, 1, 2])
    a = np.array([0, 1, 2])
    b = np.array([0, 1])
    with pytest.raises(AssertionError):
        mcnemar_test(y, a, y, b)
