from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CHANNELS = ("Fx", "Fy", "Fz", "Mz")


@dataclass(slots=True)
class SignalRun:
    run_id: str
    specimen_id: str
    sampling_rate_hz: int
    duration_seconds: float
    target_class: str
    property_value: float
    process_quality: str
    signals: dict[str, np.ndarray]


def bandpower(values: np.ndarray, sampling_rate_hz: int, low_hz: float, high_hz: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    freqs = np.fft.rfftfreq(values.size, d=1.0 / sampling_rate_hz)
    spectrum = np.abs(np.fft.rfft(values - values.mean())) ** 2
    band = (freqs >= low_hz) & (freqs < high_hz)
    if not np.any(band):
        return 0.0
    return float(spectrum[band].sum() / max(values.size, 1))


def channel_features(values: np.ndarray, sampling_rate_hz: int, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean()
    rms = float(np.sqrt(np.mean(values**2)))
    std = float(values.std())
    peak = float(np.max(np.abs(values)))
    freqs = np.fft.rfftfreq(values.size, d=1.0 / sampling_rate_hz)
    spectrum = np.abs(np.fft.rfft(centered))
    non_dc = spectrum.copy()
    if non_dc.size:
        non_dc[0] = 0.0
    spectral_mass = float(non_dc.sum())
    dominant_frequency = float(freqs[int(np.argmax(non_dc))]) if non_dc.size else 0.0
    spectral_centroid = float((freqs * non_dc).sum() / spectral_mass) if spectral_mass > 0 else 0.0
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": std,
        f"{prefix}_rms": rms,
        f"{prefix}_peak": peak,
        f"{prefix}_crest_factor": peak / max(rms, 1e-8),
        f"{prefix}_dominant_frequency_hz": dominant_frequency,
        f"{prefix}_spectral_centroid_hz": spectral_centroid,
        f"{prefix}_bandpower_low": bandpower(values, sampling_rate_hz, 0, 500),
        f"{prefix}_bandpower_mid": bandpower(values, sampling_rate_hz, 500, 2500),
        f"{prefix}_bandpower_high": bandpower(values, sampling_rate_hz, 2500, min(9000, sampling_rate_hz / 2)),
    }


def extract_signal_features(signals: dict[str, np.ndarray], sampling_rate_hz: int) -> dict[str, float]:
    missing = sorted(set(CHANNELS) - set(signals))
    if missing:
        raise ValueError(f"Missing signal channels: {missing}")

    features: dict[str, float] = {}
    for channel in CHANNELS:
        features.update(channel_features(signals[channel], sampling_rate_hz, channel))

    force_resultant = np.sqrt(signals["Fx"] ** 2 + signals["Fy"] ** 2 + signals["Fz"] ** 2)
    features.update(channel_features(force_resultant, sampling_rate_hz, "F_resultant"))
    features["specific_energy_proxy"] = float(np.mean(force_resultant * np.abs(signals["Mz"])))
    features["force_torque_corr"] = float(np.corrcoef(force_resultant, signals["Mz"])[0, 1])
    if not np.isfinite(features["force_torque_corr"]):
        features["force_torque_corr"] = 0.0
    return features

