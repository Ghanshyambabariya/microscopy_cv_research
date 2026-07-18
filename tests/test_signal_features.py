from __future__ import annotations

import numpy as np
import pandas as pd

from microscopy_cv_research.signals.features import CHANNELS, extract_signal_features
from microscopy_cv_research.signals.io import load_signal_csv
from microscopy_cv_research.signals.simulation import simulate_grinding_signals


def test_extract_signal_features_contains_expected_channels() -> None:
    sampling_rate = 2000
    t = np.arange(0, 1.0, 1.0 / sampling_rate)
    signals = {channel: np.sin(2 * np.pi * 120 * t).astype(np.float32) for channel in CHANNELS}
    features = extract_signal_features(signals, sampling_rate)
    assert "Fx_rms" in features
    assert "F_resultant_rms" in features
    assert "specific_energy_proxy" in features


def test_simulated_crack_signal_has_higher_force_than_grain() -> None:
    grain = simulate_grinding_signals("s1", "grain", 0.2, 2000, 1.0, seed=1)
    crack = simulate_grinding_signals("s2", "crack", 0.8, 2000, 1.0, seed=1)
    grain_features = extract_signal_features(grain.signals, grain.sampling_rate_hz)
    crack_features = extract_signal_features(crack.signals, crack.sampling_rate_hz)
    assert crack_features["F_resultant_rms"] > grain_features["F_resultant_rms"]


def test_load_signal_csv_requires_force_channels(tmp_path) -> None:
    path = tmp_path / "signals.csv"
    pd.DataFrame({channel: [0.0, 1.0] for channel in CHANNELS}).to_csv(path, index=False)
    signals = load_signal_csv(path)
    assert sorted(signals) == sorted(CHANNELS)
