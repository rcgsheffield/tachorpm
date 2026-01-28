# tachorpm

A Python library for extracting rotational speed (RPM) from tachometer pulse signals.

This package provides a Python implementation of [MATLAB's `tachorpm` function](https://uk.mathworks.com/help/signal/ref/tachorpm.html), commonly used in vibration analysis and rotating machinery diagnostics.

## Installation

```bash
pip install tachorpm
```

Or install from source:

```bash
git clone https://github.com/yourusername/tachorpm.git
cd tachorpm
pip install -e .
```

## Quick Start

```python
import numpy as np
from tachorpm import tachorpm

# Create a synthetic tachometer signal
fs = 1000  # Sample rate (Hz)
t = np.arange(0, 2, 1/fs)

# Simulate varying RPM (900 to 1200 RPM)
rpm_true = 900 + 150 * t
freq = rpm_true / 60  # Convert to Hz
phase = np.cumsum(2 * np.pi * freq / fs)
x = 0.5 * (1 + np.sign(np.sin(phase)))  # Square wave pulses

# Extract RPM
result = tachorpm(x, fs)

print(f"Detected {len(result.tp)} pulses")
print(f"RPM range: {result.rpm.min():.1f} - {result.rpm.max():.1f}")
```

## API Reference

### tachorpm

```python
tachorpm(x, fs, *, pulses_per_rev=1.0, state_levels=None, output_fs=None,
         fit_type='smooth', fit_points=10) -> TachoResult
```

Extract rotational speed from a tachometer pulse signal.

**Parameters:**

- `x` - Tachometer pulse signal (array-like)
- `fs` - Sample rate of the input signal (Hz)
- `pulses_per_rev` - Number of pulses per shaft revolution (default: 1.0)
- `state_levels` - Two-element array `[low, high]` for pulse detection thresholds. If `None`, levels are auto-detected using histogram analysis
- `output_fs` - Sample rate for output RPM signal (Hz). Defaults to input sample rate
- `fit_type` - Fitting method: `'smooth'` (B-spline) or `'linear'` (default: `'smooth'`)
- `fit_points` - Number of B-spline breakpoints for smooth fitting (default: 10)

**Returns:** `TachoResult` named tuple with:

- `rpm` - Rotational speed in revolutions per minute
- `t` - Time vector (seconds) corresponding to RPM values
- `tp` - Detected pulse center times (seconds)

### tachorpm_simple

```python
tachorpm_simple(x, fs, **kwargs) -> tuple[ndarray, ndarray, ndarray]
```

Convenience wrapper that returns `(rpm, t, tp)` as a tuple, matching MATLAB's output convention.

### statelevels

```python
statelevels(x, num_levels=2, histogram_bins=100) -> ndarray
```

Estimate state levels of a bilevel or multilevel signal using histogram mode analysis.

## Examples

### Basic Usage

```python
from tachorpm import tachorpm

result = tachorpm(signal, sample_rate)
rpm = result.rpm
time = result.t
pulse_times = result.tp
```

### Multiple Pulses Per Revolution

For encoders with multiple pulses per revolution:

```python
result = tachorpm(signal, fs, pulses_per_rev=60)  # 60-tooth gear
```

### Custom Output Sample Rate

Downsample the output for efficiency:

```python
result = tachorpm(signal, fs=10000, output_fs=100)  # 100 Hz output
```

### Manual State Levels

Specify thresholds for noisy signals:

```python
result = tachorpm(signal, fs, state_levels=[0.2, 4.8])  # 0-5V signal
```

### Linear Interpolation

Use linear fitting for faster processing:

```python
result = tachorpm(signal, fs, fit_type='linear')
```

## Algorithm

The algorithm performs these steps:

1. **State Level Detection** - Estimates low and high signal levels using histogram mode analysis (if not provided)

2. **Pulse Detection** - Finds rising and falling edges at the midpoint threshold, then computes pulse centers as the midpoint between each rise-fall pair

3. **RPM Calculation** - Computes instantaneous RPM from pulse timing:
   ```
   RPM = 60 / (dt * pulses_per_rev)
   ```
   where `dt` is the time between consecutive pulse centers

4. **Curve Fitting** - Fits a smooth B-spline or linear interpolation to produce the output RPM signal

## Requirements

- Python >= 3.9
- NumPy >= 1.21.0
- SciPy >= 1.7.0

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## References

1. Brandt, Anders. *Noise and Vibration Analysis: Signal Analysis and Experimental Procedures*. Chichester, UK: John Wiley & Sons, 2011.

2. Vold, Havard, and Jan Leuridan. "High Resolution Order Tracking at Extreme Slew Rates Using Kalman Tracking Filters." *Shock and Vibration*. Vol. 2, 1995, pp. 507-515.

## License

MIT License
