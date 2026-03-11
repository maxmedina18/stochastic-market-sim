from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from market_lab.validation.regime_summary import current_regime_label


def test_current_regime_label_high():
    regimes = np.array([0, 0, 1])
    assert current_regime_label(regimes) == "HIGH_VOL"


def test_current_regime_label_low():
    regimes = np.array([1, 1, 0])
    assert current_regime_label(regimes) == "LOW_VOL"


def test_current_regime_label_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        current_regime_label(np.array([]))