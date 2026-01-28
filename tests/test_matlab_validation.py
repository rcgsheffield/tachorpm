"""Tests validating tachorpm against MATLAB reference output.

These tests use data generated from MATLAB's tachorpm function to ensure
our Python implementation produces equivalent results.

MATLAB code used to generate test data:
    fs = 1000;              % 1kHz sampling
    t = (0:1/fs:10)';       % 10 seconds of data

    % Generate a frequency sweep (10Hz to 50Hz)
    % This represents 600 RPM to 3000 RPM
    freq_hz = linspace(10, 50, length(t))';
    phases = 2 * pi * cumtrapz(t, freq_hz);
    raw_signal = sin(phases);

    % Convert to a pulse train (standard tacho input)
    tacho_pulses = double(raw_signal > 0.9);

    % Get MATLAB's "Golden" reference output
    [rpm_ref, t_ref] = tachorpm(tacho_pulses, fs);
"""

from pathlib import Path

import numpy as np
import pytest

from tachorpm import tachorpm


@pytest.fixture
def matlab_test_data():
    """Load MATLAB test input and reference output."""
    data_dir = Path(__file__).parent.parent / "data"

    input_data = np.genfromtxt(
        data_dir / "test_input.csv", delimiter=",", skip_header=1
    )
    ref_data = np.genfromtxt(
        data_dir / "matlab_ref_output.csv", delimiter=",", skip_header=1
    )

    return {
        "t_input": input_data[:, 0],
        "pulse": input_data[:, 1],
        "t_ref": ref_data[:, 0],
        "rpm_ref": ref_data[:, 1],
        "fs": 1000,  # Sample rate used in MATLAB
    }


class TestMatlabValidation:
    """Tests comparing Python tachorpm output against MATLAB reference."""

    def test_output_length_matches(self, matlab_test_data):
        """Test that Python output has same length as MATLAB output."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]

        result = tachorpm(pulse, fs)

        # Allow small difference due to edge handling
        assert abs(len(result.rpm) - len(matlab_test_data["rpm_ref"])) <= 2

    def test_output_time_vector_matches(self, matlab_test_data):
        """Test that time vectors align between Python and MATLAB."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]
        t_ref = matlab_test_data["t_ref"]

        result = tachorpm(pulse, fs)

        # Time vectors should have same step size
        dt_python = result.t[1] - result.t[0]
        dt_matlab = t_ref[1] - t_ref[0]
        assert np.isclose(dt_python, dt_matlab, rtol=1e-6)

        # Time vectors should start at same point
        assert np.isclose(result.t[0], t_ref[0], atol=1e-6)

    def test_rpm_range_matches(self, matlab_test_data):
        """Test that RPM range is similar to MATLAB output."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]
        rpm_ref = matlab_test_data["rpm_ref"]

        result = tachorpm(pulse, fs)

        # RPM range should be similar (within 5%)
        python_min, python_max = result.rpm.min(), result.rpm.max()
        matlab_min, matlab_max = rpm_ref.min(), rpm_ref.max()

        assert np.isclose(python_min, matlab_min, rtol=0.05)
        assert np.isclose(python_max, matlab_max, rtol=0.05)

    def test_rpm_trend_matches(self, matlab_test_data):
        """Test that RPM increases similarly to MATLAB (frequency sweep)."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]
        rpm_ref = matlab_test_data["rpm_ref"]

        result = tachorpm(pulse, fs)

        # Check that RPM increases monotonically (it's a sweep)
        # Use a smoothed derivative to avoid noise issues
        window = 100
        rpm_smooth = np.convolve(result.rpm, np.ones(window) / window, mode="valid")
        assert np.all(np.diff(rpm_smooth) > -1)  # Allow tiny decreases from smoothing

    def test_rpm_at_key_points(self, matlab_test_data):
        """Test RPM values at specific time points match MATLAB."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]
        t_ref = matlab_test_data["t_ref"]
        rpm_ref = matlab_test_data["rpm_ref"]

        result = tachorpm(pulse, fs)

        # Test at several key time points (avoiding edges)
        # Expected: ~600 RPM at start, ~1800 RPM at middle, ~3000 RPM at end
        test_times = [1.0, 2.5, 5.0, 7.5, 9.0]

        for t_test in test_times:
            # Find closest index in both arrays
            idx_python = np.argmin(np.abs(result.t - t_test))
            idx_matlab = np.argmin(np.abs(t_ref - t_test))

            rpm_python = result.rpm[idx_python]
            rpm_matlab = rpm_ref[idx_matlab]

            # Allow 5% relative tolerance
            assert np.isclose(rpm_python, rpm_matlab, rtol=0.05), (
                f"RPM mismatch at t={t_test}s: "
                f"Python={rpm_python:.1f}, MATLAB={rpm_matlab:.1f}"
            )

    def test_rpm_correlation_with_matlab(self, matlab_test_data):
        """Test that Python RPM output is highly correlated with MATLAB."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]
        t_ref = matlab_test_data["t_ref"]
        rpm_ref = matlab_test_data["rpm_ref"]

        result = tachorpm(pulse, fs)

        # Interpolate Python result to MATLAB time points for comparison
        rpm_interp = np.interp(t_ref, result.t, result.rpm)

        # Compute correlation coefficient
        correlation = np.corrcoef(rpm_interp, rpm_ref)[0, 1]

        # Should be very highly correlated (> 0.99)
        assert correlation > 0.99, f"Correlation too low: {correlation:.4f}"

    def test_rpm_rmse_acceptable(self, matlab_test_data):
        """Test that RMSE between Python and MATLAB is acceptably low."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]
        t_ref = matlab_test_data["t_ref"]
        rpm_ref = matlab_test_data["rpm_ref"]

        result = tachorpm(pulse, fs)

        # Interpolate Python result to MATLAB time points
        rpm_interp = np.interp(t_ref, result.t, result.rpm)

        # Compute RMSE as percentage of RPM range
        rmse = np.sqrt(np.mean((rpm_interp - rpm_ref) ** 2))
        rpm_range = rpm_ref.max() - rpm_ref.min()
        rmse_pct = (rmse / rpm_range) * 100

        # RMSE should be less than 5% of the RPM range
        assert rmse_pct < 5, f"RMSE too high: {rmse_pct:.2f}% of range"

    def test_detects_correct_number_of_pulses(self, matlab_test_data):
        """Test that Python detects approximately the right number of pulses."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]

        result = tachorpm(pulse, fs)

        # For a sweep from 10Hz to 50Hz over 10 seconds,
        # integral of frequency = average frequency * time = 30Hz * 10s = 300 cycles
        # Expected pulses ~ 300
        expected_pulses = 300
        detected_pulses = len(result.tp)

        # Allow 10% tolerance
        assert abs(detected_pulses - expected_pulses) < expected_pulses * 0.1, (
            f"Pulse count mismatch: detected {detected_pulses}, "
            f"expected ~{expected_pulses}"
        )

    def test_pulse_times_within_signal_duration(self, matlab_test_data):
        """Test that all detected pulse times are within signal bounds."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]
        t_input = matlab_test_data["t_input"]

        result = tachorpm(pulse, fs)

        assert result.tp.min() >= 0, "Pulse times should be non-negative"
        assert result.tp.max() <= t_input[-1], "Pulse times should not exceed duration"

    def test_linear_fit_type(self, matlab_test_data):
        """Test that linear fit type also produces reasonable results."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]
        rpm_ref = matlab_test_data["rpm_ref"]

        result = tachorpm(pulse, fs, fit_type="linear")

        # Linear fit should still be in the right ballpark
        assert np.isclose(result.rpm.min(), rpm_ref.min(), rtol=0.1)
        assert np.isclose(result.rpm.max(), rpm_ref.max(), rtol=0.1)


class TestMatlabValidationEdgeCases:
    """Edge case tests using MATLAB validation data."""

    def test_different_output_fs(self, matlab_test_data):
        """Test with different output sample rates."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]
        rpm_ref = matlab_test_data["rpm_ref"]

        # Test with lower output sample rate
        result_100hz = tachorpm(pulse, fs, output_fs=100)

        # Should have ~1000 samples for 10s at 100Hz
        assert 990 <= len(result_100hz.rpm) <= 1010

        # RPM range should still match
        assert np.isclose(result_100hz.rpm.min(), rpm_ref.min(), rtol=0.1)
        assert np.isclose(result_100hz.rpm.max(), rpm_ref.max(), rtol=0.1)

    def test_custom_fit_points(self, matlab_test_data):
        """Test with different numbers of fit points."""
        pulse = matlab_test_data["pulse"]
        fs = matlab_test_data["fs"]
        rpm_ref = matlab_test_data["rpm_ref"]

        # More fit points should give smoother output
        result_5pts = tachorpm(pulse, fs, fit_points=5)
        result_50pts = tachorpm(pulse, fs, fit_points=50)

        # Both should be in reasonable range
        assert np.isclose(result_5pts.rpm.mean(), rpm_ref.mean(), rtol=0.1)
        assert np.isclose(result_50pts.rpm.mean(), rpm_ref.mean(), rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
