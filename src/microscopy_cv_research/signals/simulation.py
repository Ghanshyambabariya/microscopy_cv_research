from __future__ import annotations

import numpy as np

from microscopy_cv_research.signals.features import CHANNELS, SignalRun


CLASS_SIGNAL_PROFILES = {
    "grain": {"base_force": 38.0, "vibration": 0.7, "torque": 0.36, "quality_bias": "stable"},
    "pore": {"base_force": 48.0, "vibration": 1.2, "torque": 0.48, "quality_bias": "porous"},
    "crack": {"base_force": 58.0, "vibration": 2.1, "torque": 0.62, "quality_bias": "damaged"},
}


def infer_process_quality(target_class: str, property_value: float) -> str:
    if target_class == "crack" or property_value > 0.72:
        return "reject"
    if target_class == "pore" or property_value > 0.42:
        return "review"
    return "pass"


def simulate_grinding_signals(
    specimen_id: str,
    target_class: str,
    property_value: float,
    sampling_rate_hz: int,
    duration_seconds: float,
    seed: int,
) -> SignalRun:
    rng = np.random.default_rng(seed)
    profile = CLASS_SIGNAL_PROFILES.get(target_class, CLASS_SIGNAL_PROFILES["grain"])
    n = int(sampling_rate_hz * duration_seconds)
    t = np.arange(n, dtype=np.float64) / sampling_rate_hz

    spindle_hz = 120.0 + 18.0 * property_value + rng.normal(0, 1.5)
    chatter_hz = 1800.0 + 650.0 * property_value + rng.normal(0, 35.0)
    envelope = 1.0 + 0.04 * np.sin(2 * np.pi * 0.8 * t)
    burst_centers = rng.choice(n, size=max(1, int(duration_seconds * 2)), replace=False)
    burst = np.zeros(n)
    for center in burst_centers:
        width = max(8, int(0.004 * sampling_rate_hz))
        start = max(0, center - width)
        stop = min(n, center + width)
        local = np.linspace(-1, 1, stop - start)
        burst[start:stop] += np.exp(-(local * 3.0) ** 2)

    base_force = profile["base_force"] * (1.0 + 0.35 * property_value)
    vibration = profile["vibration"]
    noise = lambda scale: rng.normal(0, scale, size=n)
    fx = base_force * envelope + 2.5 * np.sin(2 * np.pi * spindle_hz * t) + vibration * np.sin(2 * np.pi * chatter_hz * t) + noise(1.2)
    fy = 0.62 * base_force * envelope + 1.8 * np.sin(2 * np.pi * (spindle_hz * 1.35) * t) + 0.7 * vibration * np.sin(2 * np.pi * chatter_hz * t) + noise(1.0)
    fz = 1.35 * base_force * envelope + 3.1 * np.sin(2 * np.pi * (spindle_hz * 0.7) * t) + 1.1 * vibration * np.sin(2 * np.pi * (chatter_hz * 0.8) * t) + noise(1.5)
    mz = profile["torque"] * envelope + 0.035 * np.sin(2 * np.pi * spindle_hz * t) + 0.02 * vibration * burst + noise(0.015)

    if target_class in {"pore", "crack"}:
        fx += burst * (2.0 + 6.0 * property_value)
        fy += burst * (1.5 + 4.0 * property_value)
        fz += burst * (3.0 + 7.0 * property_value)
    if target_class == "crack":
        crack_impulses = rng.choice(n, size=max(1, int(duration_seconds)), replace=False)
        fx[crack_impulses] += rng.normal(14, 4, size=crack_impulses.size)
        fz[crack_impulses] += rng.normal(20, 5, size=crack_impulses.size)

    signals = {channel: values.astype(np.float32) for channel, values in zip(CHANNELS, (fx, fy, fz, mz))}
    return SignalRun(
        run_id=f"grinding_{specimen_id}",
        specimen_id=specimen_id,
        sampling_rate_hz=sampling_rate_hz,
        duration_seconds=duration_seconds,
        target_class=target_class,
        property_value=property_value,
        process_quality=infer_process_quality(target_class, property_value),
        signals=signals,
    )

