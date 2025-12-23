"""Unit tests for ezmsg.baseproc.counter module."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.baseproc import (
    CounterSettings,
    CounterTransformer,
)


class TestCounterTransformer:
    """Tests for CounterTransformer."""

    def test_fixed_n_time_mode(self):
        """Test transformer with fixed n_time."""
        transformer = CounterTransformer(CounterSettings(fs=1000.0, n_time=100, mod=None))

        # Create clock tick with gain = 0.1 (10 Hz dispatch rate)
        clock_tick = AxisArray.LinearAxis(gain=0.1, offset=1.0)

        result = transformer(clock_tick)

        assert isinstance(result, AxisArray)
        assert result.data.shape == (100,)
        assert result.dims == ["time"]
        # TimeAxis has gain = 1/fs
        assert result.axes["time"].gain == 1 / 1000.0
        assert result.axes["time"].offset == 1.0  # Uses clock's offset
        np.testing.assert_array_equal(result.data, np.arange(100))

    def test_fixed_n_time_with_afap_clock(self):
        """Test transformer with fixed n_time and AFAP clock (gain=0)."""
        transformer = CounterTransformer(CounterSettings(fs=1000.0, n_time=50, mod=None))

        # AFAP clock has gain=0
        clock_tick = AxisArray.LinearAxis(gain=0.0, offset=123.456)

        result = transformer(clock_tick)

        assert isinstance(result, AxisArray)
        assert result.data.shape == (50,)
        # With AFAP clock, offset is synthetic (counter / fs)
        assert result.axes["time"].offset == 0.0  # First block starts at 0

        # Second call
        result2 = transformer(clock_tick)
        assert result2.axes["time"].offset == 50 / 1000.0  # 0.05 seconds

    def test_variable_n_time_mode(self):
        """Test transformer with n_time derived from clock gain."""
        transformer = CounterTransformer(CounterSettings(fs=1000.0, n_time=None, mod=None))

        # Clock at 10 Hz with fs=1000 -> 100 samples per tick
        clock_tick = AxisArray.LinearAxis(gain=0.1, offset=5.0)

        result = transformer(clock_tick)

        assert isinstance(result, AxisArray)
        assert result.data.shape == (100,)  # 1000 * 0.1 = 100
        assert result.axes["time"].offset == 5.0

    def test_variable_n_time_fractional_accumulation(self):
        """Test fractional sample accumulation in variable mode."""
        transformer = CounterTransformer(CounterSettings(fs=1000.0, n_time=None, mod=None))

        # Clock at 3 Hz with fs=1000 -> 333.33... samples per tick
        # Over 3 ticks we should get exactly 1000 samples (333 + 333 + 334)
        clock_tick = AxisArray.LinearAxis(gain=1.0 / 3.0, offset=0.0)

        # First tick: 333.33 -> 333 samples, 0.33 fractional
        result1 = transformer(clock_tick)
        assert result1.data.shape == (333,)

        # Second tick: 333.33 + 0.33 = 666.66 -> 333 samples, 0.66 fractional
        result2 = transformer(clock_tick)
        assert result2.data.shape == (333,)

        # Third tick: 333.33 + 0.66 = 999.99... ≈ 1000 -> 334 samples
        result3 = transformer(clock_tick)
        assert result3.data.shape == (334,)

        # Verify total is exactly 1000 (3 ticks * 333.33... = 1000)
        total_samples = sum(r.data.shape[0] for r in [result1, result2, result3])
        assert total_samples == 1000

        # Fourth tick starts fresh cycle: 333 samples
        result4 = transformer(clock_tick)
        assert result4.data.shape == (333,)

    def test_variable_n_time_returns_none_when_no_samples(self):
        """Test that transformer returns None when not enough samples accumulated."""
        transformer = CounterTransformer(
            CounterSettings(fs=10.0, n_time=None, mod=None)  # Low fs
        )

        # Clock at 100 Hz with fs=10 -> 0.1 samples per tick
        clock_tick = AxisArray.LinearAxis(gain=0.01, offset=0.0)

        # Need 10 ticks to accumulate 1 sample
        for _ in range(9):
            result = transformer(clock_tick)
            assert result is None

        # 10th tick should produce 1 sample
        result = transformer(clock_tick)
        assert result is not None
        assert result.data.shape == (1,)

    def test_variable_n_time_afap_raises_error(self):
        """Test that variable mode with AFAP clock raises error."""
        transformer = CounterTransformer(CounterSettings(fs=1000.0, n_time=None, mod=None))

        clock_tick = AxisArray.LinearAxis(gain=0.0, offset=0.0)

        with pytest.raises(ValueError, match="Cannot use clock with gain=0"):
            transformer(clock_tick)

    def test_mod_rollover(self):
        """Test counter rollover with mod."""
        transformer = CounterTransformer(CounterSettings(fs=100.0, n_time=10, mod=8))

        clock_tick = AxisArray.LinearAxis(gain=0.1, offset=0.0)

        result = transformer(clock_tick)
        np.testing.assert_array_equal(result.data, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1])

    def test_continuity_across_calls(self):
        """Test counter continuity across multiple calls."""
        transformer = CounterTransformer(CounterSettings(fs=100.0, n_time=5, mod=None))

        clock_tick = AxisArray.LinearAxis(gain=0.05, offset=0.0)

        results = [transformer(clock_tick) for _ in range(4)]
        agg = AxisArray.concatenate(*results, dim="time")
        np.testing.assert_array_equal(agg.data, np.arange(20))


class TestCounterTransformerExternalClock:
    """Tests for external clock mode patterns."""

    def test_external_clock_with_fixed_n_time(self):
        """Test external clock mode with fixed n_time."""
        transformer = CounterTransformer(CounterSettings(fs=1000.0, n_time=100, mod=None))

        # Simulate external clock ticks
        timestamps = [1.0, 1.1, 1.2, 1.3, 1.4]
        clock_ticks = [AxisArray.LinearAxis(gain=0.1, offset=ts) for ts in timestamps]

        results = [transformer(tick) for tick in clock_ticks]

        # Verify offsets match clock timestamps
        offsets = [r.axes["time"].offset for r in results]
        np.testing.assert_array_equal(offsets, timestamps)

        # Verify data continuity
        agg = AxisArray.concatenate(*results, dim="time")
        np.testing.assert_array_equal(agg.data, np.arange(500))

    def test_external_clock_variable_chunk_sizes(self):
        """Test external clock mode with variable chunk sizes."""
        transformer = CounterTransformer(CounterSettings(fs=1000.0, n_time=None, mod=None))

        # Clock with varying rates
        clock_ticks = [
            AxisArray.LinearAxis(gain=0.1, offset=0.0),  # 100 samples
            AxisArray.LinearAxis(gain=0.05, offset=0.1),  # 50 samples
            AxisArray.LinearAxis(gain=0.2, offset=0.15),  # 200 samples
        ]

        results = [transformer(tick) for tick in clock_ticks]

        assert results[0].data.shape == (100,)
        assert results[1].data.shape == (50,)
        assert results[2].data.shape == (200,)

        # Verify data continuity
        agg = AxisArray.concatenate(*results, dim="time")
        np.testing.assert_array_equal(agg.data, np.arange(350))
