"""Tests for tachorpm function."""

import numpy as np
import pytest

from tachorpm import (
    TachoResult,
    TransitionResult,
    falltime,
    risetime,
    statelevels,
    tachorpm,
    tachorpm_simple,
)


class TestStatelevels:
    """Tests for the statelevels helper function."""

    def test_bilevel_signal(self):
        """Test detection of two state levels in a square wave."""
        # Create a simple bilevel signal
        x = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        levels = statelevels(x, num_levels=2)
        assert len(levels) == 2
        assert levels[0] < levels[1]
        # Should detect levels near 0 and 1
        assert np.isclose(levels[0], 0, atol=0.2)
        assert np.isclose(levels[1], 1, atol=0.2)

    def test_noisy_bilevel_signal(self):
        """Test state level detection with noise."""
        rng = np.random.default_rng(42)
        # Create noisy bilevel signal
        x = np.concatenate(
            [
                0.1 * rng.standard_normal(100),
                1.0 + 0.1 * rng.standard_normal(100),
                0.1 * rng.standard_normal(100),
                1.0 + 0.1 * rng.standard_normal(100),
            ]
        )
        levels = statelevels(x, num_levels=2)
        assert np.isclose(levels[0], 0, atol=0.3)
        assert np.isclose(levels[1], 1, atol=0.3)

    def test_constant_signal(self):
        """Test handling of constant signal."""
        x = np.ones(100) * 5.0
        levels = statelevels(x, num_levels=2)
        assert len(levels) == 2


class TestTachorpm:
    """Tests for the main tachorpm function."""

    def test_constant_rpm(self):
        """Test extraction of constant RPM signal."""
        fs = 10000  # 10 kHz sample rate
        rpm_true = 600  # 10 Hz rotation
        duration = 2.0  # 2 seconds

        t = np.arange(0, duration, 1 / fs)
        freq = rpm_true / 60  # 10 Hz
        phase = 2 * np.pi * freq * t
        x = 0.5 * (1 + np.sign(np.sin(phase)))

        result = tachorpm(x, fs)

        assert isinstance(result, TachoResult)
        assert len(result.rpm) == len(result.t)
        assert len(result.tp) > 0

        # Check that detected RPM is close to true RPM
        # (avoid edges where extrapolation occurs)
        mid_idx = len(result.rpm) // 2
        assert np.isclose(result.rpm[mid_idx], rpm_true, rtol=0.05)

    def test_varying_rpm(self):
        """Test extraction of linearly varying RPM signal."""
        fs = 10000
        duration = 2.0
        t = np.arange(0, duration, 1 / fs)

        # Linear RPM ramp from 600 to 1200
        rpm_start, rpm_end = 600, 1200
        rpm_true = rpm_start + (rpm_end - rpm_start) * t / duration
        freq = rpm_true / 60

        # Generate variable frequency square wave
        phase = np.cumsum(2 * np.pi * freq / fs)
        x = 0.5 * (1 + np.sign(np.sin(phase)))

        result = tachorpm(x, fs, fit_type="smooth", fit_points=20)

        # Check that RPM increases over time
        assert result.rpm[-len(result.rpm) // 4] > result.rpm[len(result.rpm) // 4]

        # Check approximate range (allowing for fitting errors at edges)
        mid_start = len(result.t) // 4
        mid_end = 3 * len(result.t) // 4
        assert np.mean(result.rpm[mid_start:mid_end]) > rpm_start * 0.8
        assert np.mean(result.rpm[mid_start:mid_end]) < rpm_end * 1.2

    def test_pulses_per_rev(self):
        """Test pulses_per_rev parameter."""
        fs = 10000
        rpm_true = 600
        duration = 2.0

        t = np.arange(0, duration, 1 / fs)
        # Generate signal with 2 pulses per revolution
        freq = 2 * rpm_true / 60  # Double frequency
        phase = 2 * np.pi * freq * t
        x = 0.5 * (1 + np.sign(np.sin(phase)))

        # Without pulses_per_rev correction, RPM would be 2x
        result_wrong = tachorpm(x, fs, pulses_per_rev=1)
        result_correct = tachorpm(x, fs, pulses_per_rev=2)

        mid_idx = len(result_correct.rpm) // 2
        assert np.isclose(result_correct.rpm[mid_idx], rpm_true, rtol=0.05)
        assert np.isclose(result_wrong.rpm[mid_idx], 2 * rpm_true, rtol=0.05)

    def test_custom_state_levels(self):
        """Test custom state_levels parameter."""
        fs = 10000
        rpm_true = 600
        duration = 1.0

        t = np.arange(0, duration, 1 / fs)
        freq = rpm_true / 60
        phase = 2 * np.pi * freq * t
        # Signal oscillates between 2 and 8
        x = 5 + 3 * np.sign(np.sin(phase))

        result = tachorpm(x, fs, state_levels=[2, 8])

        mid_idx = len(result.rpm) // 2
        assert np.isclose(result.rpm[mid_idx], rpm_true, rtol=0.05)

    def test_output_fs(self):
        """Test output_fs parameter for resampling."""
        fs = 10000
        output_fs = 100
        rpm_true = 600
        duration = 2.0

        t = np.arange(0, duration, 1 / fs)
        freq = rpm_true / 60
        phase = 2 * np.pi * freq * t
        x = 0.5 * (1 + np.sign(np.sin(phase)))

        result = tachorpm(x, fs, output_fs=output_fs)

        # Output should have approximately output_fs * duration samples
        expected_samples = int(duration * output_fs) + 1
        assert abs(len(result.rpm) - expected_samples) <= 1

    def test_fit_type_linear(self):
        """Test linear fit type."""
        fs = 10000
        rpm_true = 600
        duration = 1.0

        t = np.arange(0, duration, 1 / fs)
        freq = rpm_true / 60
        phase = 2 * np.pi * freq * t
        x = 0.5 * (1 + np.sign(np.sin(phase)))

        result = tachorpm(x, fs, fit_type="linear")

        mid_idx = len(result.rpm) // 2
        assert np.isclose(result.rpm[mid_idx], rpm_true, rtol=0.05)

    def test_noisy_signal(self):
        """Test with noisy tachometer signal."""
        rng = np.random.default_rng(42)
        fs = 10000
        rpm_true = 600
        duration = 2.0

        t = np.arange(0, duration, 1 / fs)
        freq = rpm_true / 60
        phase = 2 * np.pi * freq * t
        clean = 0.5 * (1 + np.sign(np.sin(phase)))
        noise = 0.1 * rng.standard_normal(len(t))
        x = clean + noise

        result = tachorpm(x, fs)

        mid_idx = len(result.rpm) // 2
        # Allow larger tolerance for noisy signal
        assert np.isclose(result.rpm[mid_idx], rpm_true, rtol=0.1)

    def test_insufficient_pulses_error(self):
        """Test error when insufficient pulses detected."""
        fs = 1000
        # Signal with no pulses (constant) - triggers equal state levels error
        x = np.ones(1000)

        with pytest.raises(ValueError, match="Cannot detect pulses"):
            tachorpm(x, fs)

    def test_short_signal_error(self):
        """Test error for too-short signal."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            tachorpm([1.0], 1000)

    def test_invalid_fs_error(self):
        """Test error for invalid sample rate."""
        x = np.array([0, 1, 0, 1, 0, 1])
        with pytest.raises(ValueError, match="must be positive"):
            tachorpm(x, -100)

    def test_invalid_fit_type_error(self):
        """Test error for invalid fit_type."""
        fs = 10000
        t = np.arange(0, 1, 1 / fs)
        x = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 10 * t)))

        with pytest.raises(ValueError, match="fit_type"):
            tachorpm(x, fs, fit_type="invalid")

    def test_return_type(self):
        """Test that return type is TachoResult named tuple."""
        fs = 10000
        t = np.arange(0, 1, 1 / fs)
        x = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 10 * t)))

        result = tachorpm(x, fs)

        assert isinstance(result, TachoResult)
        assert hasattr(result, "rpm")
        assert hasattr(result, "t")
        assert hasattr(result, "tp")

    def test_pulse_times_in_range(self):
        """Test that detected pulse times are within signal duration."""
        fs = 10000
        duration = 1.0
        t = np.arange(0, duration, 1 / fs)
        x = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 10 * t)))

        result = tachorpm(x, fs)

        assert np.all(result.tp >= 0)
        assert np.all(result.tp <= duration)


class TestTachorpmSimple:
    """Tests for the tachorpm_simple convenience function."""

    def test_returns_tuple(self):
        """Test that tachorpm_simple returns a plain tuple."""
        fs = 10000
        t = np.arange(0, 1, 1 / fs)
        x = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 10 * t)))

        rpm, t_out, tp = tachorpm_simple(x, fs)

        assert isinstance(rpm, np.ndarray)
        assert isinstance(t_out, np.ndarray)
        assert isinstance(tp, np.ndarray)

    def test_matches_tachorpm(self):
        """Test that tachorpm_simple matches tachorpm output."""
        fs = 10000
        t = np.arange(0, 1, 1 / fs)
        x = 0.5 * (1 + np.sign(np.sin(2 * np.pi * 10 * t)))

        result = tachorpm(x, fs)
        rpm, t_out, tp = tachorpm_simple(x, fs)

        np.testing.assert_array_equal(rpm, result.rpm)
        np.testing.assert_array_equal(t_out, result.t)
        np.testing.assert_array_equal(tp, result.tp)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_period(self):
        """Test with minimal number of pulses."""
        fs = 10000
        rpm_true = 600
        # Need 3.5 periods because signal starts at threshold and first edge is missed
        duration = 3.5 / (rpm_true / 60)

        t = np.arange(0, duration, 1 / fs)
        freq = rpm_true / 60
        phase = 2 * np.pi * freq * t
        x = 0.5 * (1 + np.sign(np.sin(phase)))

        result = tachorpm(x, fs)
        assert len(result.tp) >= 2

    def test_high_rpm(self):
        """Test with high RPM (high frequency pulses)."""
        fs = 100000  # 100 kHz to capture high frequency
        rpm_true = 30000  # 500 Hz
        duration = 0.1

        t = np.arange(0, duration, 1 / fs)
        freq = rpm_true / 60
        phase = 2 * np.pi * freq * t
        x = 0.5 * (1 + np.sign(np.sin(phase)))

        result = tachorpm(x, fs)

        mid_idx = len(result.rpm) // 2
        assert np.isclose(result.rpm[mid_idx], rpm_true, rtol=0.05)

    def test_low_rpm(self):
        """Test with low RPM (low frequency pulses)."""
        fs = 1000
        rpm_true = 30  # 0.5 Hz
        duration = 10.0  # Need longer duration to get enough pulses

        t = np.arange(0, duration, 1 / fs)
        freq = rpm_true / 60
        phase = 2 * np.pi * freq * t
        x = 0.5 * (1 + np.sign(np.sin(phase)))

        result = tachorpm(x, fs)

        mid_idx = len(result.rpm) // 2
        assert np.isclose(result.rpm[mid_idx], rpm_true, rtol=0.1)

    def test_asymmetric_duty_cycle(self):
        """Test with non-50% duty cycle pulse."""
        fs = 10000
        rpm_true = 600
        duration = 2.0

        t = np.arange(0, duration, 1 / fs)
        freq = rpm_true / 60
        phase = 2 * np.pi * freq * t
        # 25% duty cycle (high when sin > 0.5)
        x = (np.sin(phase) > 0.5).astype(float)

        result = tachorpm(x, fs)

        # Should still detect correct RPM
        mid_idx = len(result.rpm) // 2
        assert np.isclose(result.rpm[mid_idx], rpm_true, rtol=0.1)


class TestRisetime:
    """Tests for the risetime function."""

    def test_single_ramp(self):
        """Test risetime on a single linear ramp."""
        fs = 10000
        t = np.arange(0, 0.1, 1 / fs)
        # Ramp from 0 to 1 over 10ms starting at t=20ms
        x = np.where(t < 0.02, 0.0, np.where(t < 0.03, (t - 0.02) * 100, 1.0))

        result = risetime(x, fs)

        assert isinstance(result, TransitionResult)
        assert len(result.duration) == 1
        # 10% to 90% of 10ms ramp = 8ms
        assert np.isclose(result.duration[0], 0.008, rtol=0.05)

    def test_multiple_transitions(self):
        """Test risetime with multiple rising transitions."""
        fs = 10000
        duration = 0.2
        t = np.arange(0, duration, 1 / fs)
        # Square wave with finite rise time (using tanh for smooth transitions)
        freq = 20  # 20 Hz
        phase = 2 * np.pi * freq * t
        # Create signal with gradual transitions
        x = 0.5 + 0.5 * np.tanh(10 * np.sin(phase))

        result = risetime(x, fs)

        # Should detect multiple rising transitions
        assert len(result.duration) >= 2

    def test_custom_reference_levels(self):
        """Test risetime with custom percent reference levels."""
        fs = 10000
        t = np.arange(0, 0.1, 1 / fs)
        # Linear ramp from 0 to 1 over 10ms
        x = np.where(t < 0.02, 0.0, np.where(t < 0.03, (t - 0.02) * 100, 1.0))

        # 20% to 80% reference levels (60% of the 10ms ramp = 6ms)
        result = risetime(x, fs, percent_reference_levels=(20.0, 80.0))

        assert len(result.duration) == 1
        assert np.isclose(result.duration[0], 0.006, rtol=0.05)

    def test_custom_state_levels(self):
        """Test risetime with explicit state levels."""
        fs = 10000
        t = np.arange(0, 0.1, 1 / fs)
        # Ramp from 2 to 8 over 10ms
        x = np.where(t < 0.02, 2.0, np.where(t < 0.03, 2.0 + (t - 0.02) * 600, 8.0))

        result = risetime(x, fs, state_levels=[2, 8])

        assert len(result.duration) == 1
        # 10% to 90% of 10ms ramp = 8ms
        assert np.isclose(result.duration[0], 0.008, rtol=0.05)

    def test_reference_levels_returned(self):
        """Test that reference levels are correctly returned."""
        fs = 10000
        t = np.arange(0, 0.1, 1 / fs)
        x = np.where(t < 0.02, 0.0, np.where(t < 0.03, (t - 0.02) * 100, 1.0))

        result = risetime(x, fs, state_levels=[0, 1])

        # 10% of 0-1 range = 0.1, 90% = 0.9
        assert np.isclose(result.initial_level, 0.1, rtol=0.01)
        assert np.isclose(result.final_level, 0.9, rtol=0.01)

    def test_crossing_times(self):
        """Test that crossing times are reasonable."""
        fs = 10000
        t = np.arange(0, 0.1, 1 / fs)
        # Ramp starts at t=0.02, ends at t=0.03
        x = np.where(t < 0.02, 0.0, np.where(t < 0.03, (t - 0.02) * 100, 1.0))

        result = risetime(x, fs, state_levels=[0, 1])

        assert len(result.initial_cross) == 1
        assert len(result.final_cross) == 1
        # 10% crossing at t = 0.02 + 0.001 = 0.021
        assert np.isclose(result.initial_cross[0], 0.021, atol=0.001)
        # 90% crossing at t = 0.02 + 0.009 = 0.029
        assert np.isclose(result.final_cross[0], 0.029, atol=0.001)

    def test_no_transitions(self):
        """Test risetime with no rising transitions."""
        fs = 10000
        # Constant signal
        x = np.ones(1000)

        result = risetime(x, fs, state_levels=[0, 1])

        assert len(result.duration) == 0
        assert len(result.initial_cross) == 0
        assert len(result.final_cross) == 0

    def test_invalid_reference_levels(self):
        """Test error for invalid reference level order."""
        fs = 10000
        x = np.linspace(0, 1, 1000)

        with pytest.raises(ValueError, match="percent_reference_levels"):
            risetime(x, fs, percent_reference_levels=(90.0, 10.0))


class TestFalltime:
    """Tests for the falltime function."""

    def test_single_ramp(self):
        """Test falltime on a single falling ramp."""
        fs = 10000
        t = np.arange(0, 0.1, 1 / fs)
        # Ramp from 1 to 0 over 10ms starting at t=20ms
        x = np.where(t < 0.02, 1.0, np.where(t < 0.03, 1.0 - (t - 0.02) * 100, 0.0))

        result = falltime(x, fs)

        assert isinstance(result, TransitionResult)
        assert len(result.duration) == 1
        # 90% to 10% of 10ms ramp = 8ms
        assert np.isclose(result.duration[0], 0.008, rtol=0.05)

    def test_multiple_transitions(self):
        """Test falltime with multiple falling transitions."""
        fs = 10000
        duration = 0.2
        t = np.arange(0, duration, 1 / fs)
        # Square wave with finite fall time
        freq = 20  # 20 Hz
        phase = 2 * np.pi * freq * t
        x = 0.5 + 0.5 * np.tanh(10 * np.sin(phase))

        result = falltime(x, fs)

        # Should detect multiple falling transitions
        assert len(result.duration) >= 2

    def test_reference_levels_for_fall(self):
        """Test that reference levels are swapped for falling edge."""
        fs = 10000
        t = np.arange(0, 0.1, 1 / fs)
        x = np.where(t < 0.02, 1.0, np.where(t < 0.03, 1.0 - (t - 0.02) * 100, 0.0))

        result = falltime(x, fs, state_levels=[0, 1])

        # For falling: initial is 90% (0.9), final is 10% (0.1)
        assert np.isclose(result.initial_level, 0.9, rtol=0.01)
        assert np.isclose(result.final_level, 0.1, rtol=0.01)

    def test_crossing_times_fall(self):
        """Test crossing times for falling transition."""
        fs = 10000
        t = np.arange(0, 0.1, 1 / fs)
        # Ramp down starts at t=0.02, ends at t=0.03
        x = np.where(t < 0.02, 1.0, np.where(t < 0.03, 1.0 - (t - 0.02) * 100, 0.0))

        result = falltime(x, fs, state_levels=[0, 1])

        assert len(result.initial_cross) == 1
        assert len(result.final_cross) == 1
        # Initial (90%) crossing at t = 0.02 + 0.001 = 0.021
        assert np.isclose(result.initial_cross[0], 0.021, atol=0.001)
        # Final (10%) crossing at t = 0.02 + 0.009 = 0.029
        assert np.isclose(result.final_cross[0], 0.029, atol=0.001)


class TestRisetimeFalltimeSymmetry:
    """Tests for symmetric behavior between risetime and falltime."""

    def test_symmetric_trapezoid(self):
        """Test that rise and fall times are equal for symmetric trapezoid."""
        fs = 10000
        t = np.arange(0, 0.15, 1 / fs)
        # Symmetric trapezoid: rise 10ms, hold 30ms, fall 10ms
        x = np.where(
            t < 0.02,
            0.0,
            np.where(
                t < 0.03,
                (t - 0.02) * 100,  # Rise
                np.where(
                    t < 0.08,
                    1.0,  # Hold high
                    np.where(t < 0.09, 1.0 - (t - 0.08) * 100, 0.0),  # Fall
                ),
            ),
        )

        rise_result = risetime(x, fs, state_levels=[0, 1])
        fall_result = falltime(x, fs, state_levels=[0, 1])

        assert len(rise_result.duration) == 1
        assert len(fall_result.duration) == 1
        # Both should be 8ms (10% to 90% of 10ms)
        assert np.isclose(rise_result.duration[0], fall_result.duration[0], rtol=0.05)

    def test_asymmetric_transitions(self):
        """Test different rise and fall times."""
        fs = 10000
        t = np.arange(0, 0.15, 1 / fs)
        # Asymmetric: fast rise (5ms), slow fall (20ms)
        x = np.where(
            t < 0.02,
            0.0,
            np.where(
                t < 0.025,
                (t - 0.02) * 200,  # Fast rise (5ms)
                np.where(
                    t < 0.06,
                    1.0,  # Hold high
                    np.where(t < 0.08, 1.0 - (t - 0.06) * 50, 0.0),  # Slow fall (20ms)
                ),
            ),
        )

        rise_result = risetime(x, fs, state_levels=[0, 1])
        fall_result = falltime(x, fs, state_levels=[0, 1])

        # Rise: 80% of 5ms = 4ms
        assert np.isclose(rise_result.duration[0], 0.004, rtol=0.1)
        # Fall: 80% of 20ms = 16ms
        assert np.isclose(fall_result.duration[0], 0.016, rtol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
