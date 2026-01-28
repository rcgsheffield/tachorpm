"""
tachorpm - Extract rotational speed from tachometer pulse signal.

This module provides a Python implementation of MATLAB's tachorpm function
for extracting RPM (rotations per minute) from tachometer pulse signals.

References
----------
[1] Brandt, Anders. Noise and Vibration Analysis: Signal Analysis and
    Experimental Procedures. Chichester, UK: John Wiley & Sons, 2011.

[2] Vold, Håvard, and Jan Leuridan. "High Resolution Order Tracking at
    Extreme Slew Rates Using Kalman Tracking Filters." Shock and Vibration.
    Vol. 2, 1995, pp. 507–515.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import interpolate


class TachoResult(NamedTuple):
    """Result container for tachorpm function.

    Attributes
    ----------
    rpm : ndarray
        Rotational speed in revolutions per minute, resampled at output_fs.
    t : ndarray
        Time vector in seconds corresponding to rpm values.
    tp : ndarray
        Detected pulse center locations in seconds.
    """

    rpm: NDArray[np.floating]
    t: NDArray[np.floating]
    tp: NDArray[np.floating]


def statelevels(
    x: ArrayLike,
    num_levels: int = 2,
    histogram_bins: int = 100,
) -> NDArray[np.floating]:
    """Estimate state levels of a bilevel or multilevel signal.

    Uses histogram mode analysis to find the most probable signal levels,
    similar to MATLAB's statelevels function.

    Parameters
    ----------
    x : array_like
        Input signal.
    num_levels : int, default=2
        Number of state levels to detect.
    histogram_bins : int, default=100
        Number of histogram bins for analysis.

    Returns
    -------
    levels : ndarray
        Detected state levels in ascending order.
    """
    x = np.asarray(x).ravel()
    x_min, x_max = np.min(x), np.max(x)

    if x_min == x_max:
        return np.array([x_min, x_max])

    # Create histogram
    counts, bin_edges = np.histogram(x, bins=histogram_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find peaks in histogram (state levels are modes)
    # Simple approach: divide into num_levels regions and find max in each
    levels = np.zeros(num_levels)
    region_size = histogram_bins // num_levels

    for i in range(num_levels):
        start_idx = i * region_size
        end_idx = (i + 1) * region_size if i < num_levels - 1 else histogram_bins
        region_counts = counts[start_idx:end_idx]
        max_idx = np.argmax(region_counts) + start_idx
        levels[i] = bin_centers[max_idx]

    return np.sort(levels)


def _find_crossings(
    x: NDArray[np.floating],
    t: NDArray[np.floating],
    threshold: float,
    edge: Literal["rising", "falling"],
    hysteresis: float = 0.0,
) -> NDArray[np.floating]:
    """Find threshold crossing times with linear interpolation.

    Parameters
    ----------
    x : ndarray
        Input signal.
    t : ndarray
        Time vector.
    threshold : float
        Crossing threshold level.
    edge : {'rising', 'falling'}
        Type of edge to detect.
    hysteresis : float, default=0.0
        Hysteresis band to avoid false crossings from noise.

    Returns
    -------
    crossings : ndarray
        Times of threshold crossings.
    """
    if edge == "rising":
        # Rising edge: signal goes from below to above threshold
        below = x[:-1] < threshold - hysteresis
        above = x[1:] >= threshold + hysteresis
        crossing_mask = below & above
    else:
        # Falling edge: signal goes from above to below threshold
        above = x[:-1] > threshold + hysteresis
        below = x[1:] <= threshold - hysteresis
        crossing_mask = above & below

    crossing_indices = np.where(crossing_mask)[0]

    if len(crossing_indices) == 0:
        return np.array([])

    # Linear interpolation to find exact crossing times
    x0 = x[crossing_indices]
    x1 = x[crossing_indices + 1]
    t0 = t[crossing_indices]
    t1 = t[crossing_indices + 1]

    # Avoid division by zero
    dx = x1 - x0
    dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)

    # Interpolate crossing time
    alpha = (threshold - x0) / dx
    crossings = t0 + alpha * (t1 - t0)

    return crossings


def _detect_pulse_centers(
    x: NDArray[np.floating],
    t: NDArray[np.floating],
    low_level: float,
    high_level: float,
    tolerance: float = 0.1,
) -> NDArray[np.floating]:
    """Detect pulse center times using rise and fall edge detection.

    The pulse center is computed as the midpoint between each rising edge
    and the subsequent falling edge.

    Parameters
    ----------
    x : ndarray
        Input tachometer signal.
    t : ndarray
        Time vector.
    low_level : float
        Low state level.
    high_level : float
        High state level.
    tolerance : float, default=0.1
        Tolerance band (fraction of amplitude) for edge detection.

    Returns
    -------
    pulse_centers : ndarray
        Times of pulse centers in seconds.
    """
    amplitude = high_level - low_level
    # Use midpoint threshold with hysteresis for robust detection
    mid_threshold = (low_level + high_level) / 2
    hysteresis = tolerance * amplitude

    # Find rising and falling edges
    rising_times = _find_crossings(x, t, mid_threshold, "rising", hysteresis)
    falling_times = _find_crossings(x, t, mid_threshold, "falling", hysteresis)

    if len(rising_times) == 0 or len(falling_times) == 0:
        return np.array([])

    # Match each rising edge with the next falling edge
    pulse_centers = []
    fall_idx = 0

    for rise_time in rising_times:
        # Find the next falling edge after this rising edge
        while fall_idx < len(falling_times) and falling_times[fall_idx] <= rise_time:
            fall_idx += 1

        if fall_idx >= len(falling_times):
            break

        # Pulse center is midpoint between rise and fall
        center = (rise_time + falling_times[fall_idx]) / 2
        pulse_centers.append(center)
        fall_idx += 1

    return np.array(pulse_centers)


def _fit_rpm_smooth(
    t_rpm: NDArray[np.floating],
    rpm_raw: NDArray[np.floating],
    t_output: NDArray[np.floating],
    fit_points: int,
) -> NDArray[np.floating]:
    """Fit RPM data using least-squares B-spline smoothing.

    Parameters
    ----------
    t_rpm : ndarray
        Time points of raw RPM measurements.
    rpm_raw : ndarray
        Raw RPM values.
    t_output : ndarray
        Desired output time vector.
    fit_points : int
        Number of spline breakpoints (knots).

    Returns
    -------
    rpm_fitted : ndarray
        Smoothed RPM values at output times.
    """
    if len(t_rpm) < 4:
        # Not enough points for spline, fall back to linear
        return _fit_rpm_linear(t_rpm, rpm_raw, t_output)

    # Create interior knots for B-spline
    # Ensure knots are within the data range
    t_min, t_max = t_rpm[0], t_rpm[-1]
    num_interior_knots = max(1, fit_points - 2)
    interior_knots = np.linspace(t_min, t_max, num_interior_knots + 2)[1:-1]

    try:
        # Fit least-squares B-spline (cubic, k=3)
        tck = interpolate.splrep(t_rpm, rpm_raw, t=interior_knots, k=3)
        rpm_fitted = interpolate.splev(t_output, tck)
    except Exception:
        # Fall back to linear interpolation if spline fails
        rpm_fitted = _fit_rpm_linear(t_rpm, rpm_raw, t_output)

    return np.asarray(rpm_fitted)


def _fit_rpm_linear(
    t_rpm: NDArray[np.floating],
    rpm_raw: NDArray[np.floating],
    t_output: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Fit RPM data using linear interpolation.

    Parameters
    ----------
    t_rpm : ndarray
        Time points of raw RPM measurements.
    rpm_raw : ndarray
        Raw RPM values.
    t_output : ndarray
        Desired output time vector.

    Returns
    -------
    rpm_fitted : ndarray
        Interpolated RPM values at output times.
    """
    if len(t_rpm) < 2:
        # Single point or empty - return constant
        if len(t_rpm) == 1:
            return np.full_like(t_output, rpm_raw[0])
        return np.zeros_like(t_output)

    # Linear interpolation with extrapolation
    rpm_fitted = np.interp(t_output, t_rpm, rpm_raw)
    return rpm_fitted


def tachorpm(
    x: ArrayLike,
    fs: float,
    *,
    pulses_per_rev: float = 1.0,
    state_levels: ArrayLike | None = None,
    output_fs: float | None = None,
    fit_type: Literal["smooth", "linear"] = "smooth",
    fit_points: int = 10,
) -> TachoResult:
    """Extract rotational speed from tachometer pulse signal.

    This function processes a tachometer pulse signal to extract the
    rotational speed (RPM) over time. It detects pulse edges, computes
    instantaneous RPM from pulse timing, and fits a smooth or linear
    curve to the resulting RPM signal.

    Parameters
    ----------
    x : array_like
        Tachometer pulse signal vector.
    fs : float
        Sample rate of the input signal in Hz.
    pulses_per_rev : float, default=1.0
        Number of tachometer pulses per revolution of the rotating shaft.
    state_levels : array_like or None, default=None
        Two-element array [low, high] specifying the state level thresholds
        for pulse detection. If None, levels are automatically computed
        using histogram analysis.
    output_fs : float or None, default=None
        Sample rate for the output RPM signal in Hz. If None, uses the
        input sample rate fs.
    fit_type : {'smooth', 'linear'}, default='smooth'
        Method for fitting the RPM signal:
        - 'smooth': Least-squares B-spline fitting for smooth output
        - 'linear': Linear interpolation between measurement points
    fit_points : int, default=10
        Number of B-spline breakpoints for smooth fitting. Higher values
        allow the fit to follow rapid RPM changes more closely. Only used
        when fit_type='smooth'.

    Returns
    -------
    result : TachoResult
        Named tuple containing:
        - rpm: Rotational speed in revolutions per minute
        - t: Time vector in seconds corresponding to rpm
        - tp: Detected pulse center times in seconds

    Raises
    ------
    ValueError
        If input signal is too short, sample rate is invalid, or
        insufficient pulses are detected.

    Notes
    -----
    The algorithm performs these steps:

    1. **State Level Detection**: If not provided, automatically estimates
       the low and high signal levels using histogram mode analysis.

    2. **Pulse Detection**: Finds rising and falling edges where the signal
       crosses the midpoint threshold between state levels. Pulse centers
       are computed as the midpoint between each rise-fall pair.

    3. **RPM Calculation**: Computes instantaneous RPM from the time
       difference between consecutive pulses:

       .. math::

           \\text{RPM} = \\frac{60}{\\Delta t \\cdot \\text{pulses\\_per\\_rev}}

       where :math:`\\Delta t` is the time between consecutive pulse centers.

    4. **Curve Fitting**: Fits either a smooth B-spline or linear
       interpolation to produce the output RPM signal at the specified
       sample rate.

    Examples
    --------
    Generate a synthetic tachometer signal and extract RPM:

    >>> import numpy as np
    >>> fs = 1000  # 1 kHz sample rate
    >>> t = np.arange(0, 2, 1/fs)
    >>> # Simulate varying RPM (900 to 1200 RPM)
    >>> rpm_true = 900 + 150 * t
    >>> freq = rpm_true / 60  # Hz
    >>> phase = np.cumsum(2 * np.pi * freq / fs)
    >>> x = 0.5 * (1 + np.sign(np.sin(phase)))  # Square wave
    >>> result = tachorpm(x, fs)
    >>> print(f"Detected {len(result.tp)} pulses")

    Extract RPM with custom parameters:

    >>> result = tachorpm(
    ...     x, fs,
    ...     pulses_per_rev=2,
    ...     fit_type='linear',
    ...     output_fs=100
    ... )

    References
    ----------
    .. [1] Brandt, Anders. Noise and Vibration Analysis: Signal Analysis
           and Experimental Procedures. Chichester, UK: John Wiley & Sons,
           2011.

    .. [2] Vold, Håvard, and Jan Leuridan. "High Resolution Order Tracking
           at Extreme Slew Rates Using Kalman Tracking Filters." Shock and
           Vibration. Vol. 2, 1995, pp. 507–515.
    """
    # Validate and convert inputs
    x = np.asarray(x, dtype=np.float64).ravel()
    n_samples = len(x)

    if n_samples < 2:
        raise ValueError("Input signal must have at least 2 samples")

    if fs <= 0:
        raise ValueError("Sample rate fs must be positive")

    if pulses_per_rev <= 0:
        raise ValueError("pulses_per_rev must be positive")

    if output_fs is None:
        output_fs = fs
    elif output_fs <= 0:
        raise ValueError("output_fs must be positive")

    if fit_type not in ("smooth", "linear"):
        raise ValueError("fit_type must be 'smooth' or 'linear'")

    if fit_points < 1:
        raise ValueError("fit_points must be at least 1")

    # Create time vector for input signal
    t_input = np.arange(n_samples) / fs

    # Determine state levels
    if state_levels is None:
        levels = statelevels(x, num_levels=2)
        low_level, high_level = levels[0], levels[1]
    else:
        state_levels = np.asarray(state_levels).ravel()
        if len(state_levels) != 2:
            raise ValueError("state_levels must have exactly 2 elements")
        low_level, high_level = np.sort(state_levels)

    if high_level <= low_level:
        raise ValueError(
            "Cannot detect pulses: high and low state levels are equal. "
            "Check input signal or provide explicit state_levels."
        )

    # Detect pulse centers
    tp = _detect_pulse_centers(x, t_input, low_level, high_level)

    if len(tp) < 2:
        raise ValueError(
            f"Insufficient pulses detected ({len(tp)}). "
            "Need at least 2 pulses to compute RPM. "
            "Try adjusting state_levels parameter."
        )

    # Compute instantaneous RPM from pulse timing
    # Time between consecutive pulses
    dt = np.diff(tp)

    # RPM at midpoint between each pair of pulses
    t_rpm = (tp[:-1] + tp[1:]) / 2
    rpm_raw = 60.0 / (dt * pulses_per_rev)

    # Create output time vector
    duration = t_input[-1]
    n_output = int(np.ceil(duration * output_fs)) + 1
    t_output = np.arange(n_output) / output_fs

    # Clip to valid range
    t_output = t_output[t_output <= duration]

    # Fit RPM curve to output time grid
    if fit_type == "smooth":
        rpm_output = _fit_rpm_smooth(t_rpm, rpm_raw, t_output, fit_points)
    else:
        rpm_output = _fit_rpm_linear(t_rpm, rpm_raw, t_output)

    return TachoResult(rpm=rpm_output, t=t_output, tp=tp)


# Convenience function that matches MATLAB's calling convention more closely
def tachorpm_simple(
    x: ArrayLike,
    fs: float,
    **kwargs,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Extract rotational speed from tachometer pulse signal (tuple output).

    This is a convenience wrapper around tachorpm() that returns a tuple
    instead of a named tuple, matching MATLAB's output convention.

    See tachorpm() for full documentation.

    Returns
    -------
    rpm : ndarray
        Rotational speed in revolutions per minute.
    t : ndarray
        Time vector in seconds.
    tp : ndarray
        Detected pulse locations in seconds.
    """
    result = tachorpm(x, fs, **kwargs)
    return result.rpm, result.t, result.tp
