"""Unit tests for ezmsg.baseproc.clockdriven module."""

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.baseproc import (
    BaseClockDrivenProducer,
    ClockDrivenSettings,
    ClockDrivenState,
    processor_state,
)

# --- Test implementations ---


class CounterProducerSettings(ClockDrivenSettings):
    """Simple counter producer settings for testing."""

    mod: int | None = None
    """If set, counter values rollover at this modulus."""


class CounterProducer(BaseClockDrivenProducer[CounterProducerSettings, ClockDrivenState]):
    """
    Simple counter producer for testing.

    Outputs AxisArray with monotonically increasing counter values.
    This is similar to CounterTransformer but uses BaseClockDrivenProducer.
    """

    def _reset_state(self, time_axis: AxisArray.LinearAxis) -> None:
        """Reset state - nothing special needed for counter."""
        pass

    def _produce(self, n_samples: int, time_axis: AxisArray.LinearAxis) -> AxisArray:
        """Generate counter values."""
        # Generate counter data (using pre-increment counter value)
        block_samp = np.arange(self._state.counter, self._state.counter + n_samples)
        if self.settings.mod is not None:
            block_samp = block_samp % self.settings.mod

        return AxisArray(
            data=block_samp,
            dims=["time"],
            axes={"time": time_axis},
            key="counter",
        )


@processor_state
class SinProducerState(ClockDrivenState):
    """State for sine wave producer."""

    ang_freq: float = 0.0
    amp: float = 1.0
    phase: float = 0.0


class SinProducerSettings(ClockDrivenSettings):
    """Sine wave producer settings."""

    freq: float = 1.0
    """Frequency in Hz."""

    amp: float = 1.0
    """Amplitude."""

    phase: float = 0.0
    """Initial phase in radians."""


class SinProducer(BaseClockDrivenProducer[SinProducerSettings, SinProducerState]):
    """
    Sine wave producer for testing.

    Demonstrates a more complex clock-driven producer with custom state.
    """

    def _reset_state(self, time_axis: AxisArray.LinearAxis) -> None:
        """Pre-compute angular frequency."""
        self._state.ang_freq = 2 * np.pi * self.settings.freq
        self._state.amp = self.settings.amp
        self._state.phase = self.settings.phase

    def _produce(self, n_samples: int, time_axis: AxisArray.LinearAxis) -> AxisArray:
        """Generate sine wave data."""
        # Calculate time values for this chunk
        t = (np.arange(n_samples) + self._state.counter) * time_axis.gain
        data = self._state.amp * np.sin(self._state.ang_freq * t + self._state.phase)

        return AxisArray(
            data=data,
            dims=["time"],
            axes={"time": time_axis},
            key="sine",
        )


# --- Tests ---


class TestBaseClockDrivenProducer:
    """Tests for BaseClockDrivenProducer."""

    def test_fixed_n_time_mode(self):
        """Test producer with fixed n_time."""
        producer = CounterProducer(CounterProducerSettings(fs=1000.0, n_time=100, mod=None))

        # Create clock tick with gain = 0.1 (10 Hz dispatch rate)
        clock_tick = AxisArray.LinearAxis(gain=0.1, offset=1.0)

        result = producer(clock_tick)

        assert isinstance(result, AxisArray)
        assert result.data.shape == (100,)
        assert result.dims == ["time"]
        # TimeAxis has gain = 1/fs
        assert result.axes["time"].gain == 1 / 1000.0
        assert result.axes["time"].offset == 1.0  # Uses clock's offset
        np.testing.assert_array_equal(result.data, np.arange(100))

    def test_fixed_n_time_with_afap_clock(self):
        """Test producer with fixed n_time and AFAP clock (gain=0)."""
        producer = CounterProducer(CounterProducerSettings(fs=1000.0, n_time=50, mod=None))

        # AFAP clock has gain=0
        clock_tick = AxisArray.LinearAxis(gain=0.0, offset=123.456)

        result = producer(clock_tick)

        assert isinstance(result, AxisArray)
        assert result.data.shape == (50,)
        # With AFAP clock, offset is synthetic (counter / fs)
        assert result.axes["time"].offset == 0.0  # First block starts at 0

        # Second call
        result2 = producer(clock_tick)
        assert result2.axes["time"].offset == 50 / 1000.0  # 0.05 seconds

    def test_variable_n_time_mode(self):
        """Test producer with n_time derived from clock gain."""
        producer = CounterProducer(CounterProducerSettings(fs=1000.0, n_time=None, mod=None))

        # Clock at 10 Hz with fs=1000 -> 100 samples per tick
        clock_tick = AxisArray.LinearAxis(gain=0.1, offset=5.0)

        result = producer(clock_tick)

        assert isinstance(result, AxisArray)
        assert result.data.shape == (100,)  # 1000 * 0.1 = 100
        assert result.axes["time"].offset == 5.0

    def test_variable_n_time_fractional_accumulation(self):
        """Test fractional sample accumulation in variable mode."""
        producer = CounterProducer(CounterProducerSettings(fs=1000.0, n_time=None, mod=None))

        # Clock at 3 Hz with fs=1000 -> 333.33... samples per tick
        # Over 3 ticks we should get exactly 1000 samples (333 + 333 + 334)
        clock_tick = AxisArray.LinearAxis(gain=1.0 / 3.0, offset=0.0)

        # First tick: 333.33 -> 333 samples, 0.33 fractional
        result1 = producer(clock_tick)
        assert result1.data.shape == (333,)

        # Second tick: 333.33 + 0.33 = 666.66 -> 333 samples, 0.66 fractional
        result2 = producer(clock_tick)
        assert result2.data.shape == (333,)

        # Third tick: 333.33 + 0.66 = 999.99... ≈ 1000 -> 334 samples
        result3 = producer(clock_tick)
        assert result3.data.shape == (334,)

        # Verify total is exactly 1000 (3 ticks * 333.33... = 1000)
        total_samples = sum(r.data.shape[0] for r in [result1, result2, result3])
        assert total_samples == 1000

        # Fourth tick starts fresh cycle: 333 samples
        result4 = producer(clock_tick)
        assert result4.data.shape == (333,)

    def test_variable_n_time_returns_none_when_no_samples(self):
        """Test that producer returns None when not enough samples accumulated."""
        producer = CounterProducer(
            CounterProducerSettings(fs=10.0, n_time=None, mod=None)  # Low fs
        )

        # Clock at 100 Hz with fs=10 -> 0.1 samples per tick
        clock_tick = AxisArray.LinearAxis(gain=0.01, offset=0.0)

        # Need 10 ticks to accumulate 1 sample
        for _ in range(9):
            result = producer(clock_tick)
            assert result is None

        # 10th tick should produce 1 sample
        result = producer(clock_tick)
        assert result is not None
        assert result.data.shape == (1,)

    def test_variable_n_time_afap_raises_error(self):
        """Test that variable mode with AFAP clock raises error."""
        producer = CounterProducer(CounterProducerSettings(fs=1000.0, n_time=None, mod=None))

        clock_tick = AxisArray.LinearAxis(gain=0.0, offset=0.0)

        with pytest.raises(ValueError, match="Cannot use clock with gain=0"):
            producer(clock_tick)

    def test_mod_rollover(self):
        """Test counter rollover with mod."""
        producer = CounterProducer(CounterProducerSettings(fs=100.0, n_time=10, mod=8))

        clock_tick = AxisArray.LinearAxis(gain=0.1, offset=0.0)

        result = producer(clock_tick)
        np.testing.assert_array_equal(result.data, [0, 1, 2, 3, 4, 5, 6, 7, 0, 1])

    def test_continuity_across_calls(self):
        """Test counter continuity across multiple calls."""
        producer = CounterProducer(CounterProducerSettings(fs=100.0, n_time=5, mod=None))

        clock_tick = AxisArray.LinearAxis(gain=0.05, offset=0.0)

        results = [producer(clock_tick) for _ in range(4)]
        agg = AxisArray.concatenate(*results, dim="time")
        np.testing.assert_array_equal(agg.data, np.arange(20))


class TestSinProducer:
    """Tests for the example SinProducer implementation."""

    def test_basic_sine_generation(self):
        """Test basic sine wave generation."""
        fs = 1000.0
        freq = 10.0
        producer = SinProducer(SinProducerSettings(fs=fs, n_time=100, freq=freq, amp=1.0, phase=0.0))

        clock_tick = AxisArray.LinearAxis(gain=0.1, offset=0.0)

        result = producer(clock_tick)

        assert isinstance(result, AxisArray)
        assert result.data.shape == (100,)

        # Verify it's actually a sine wave
        t = np.arange(100) / fs
        expected = np.sin(2 * np.pi * freq * t)
        np.testing.assert_allclose(result.data, expected, rtol=1e-10)

    def test_sine_continuity_across_chunks(self):
        """Test that sine wave is continuous across multiple chunks."""
        fs = 1000.0
        freq = 10.0
        producer = SinProducer(SinProducerSettings(fs=fs, n_time=50, freq=freq, amp=1.0, phase=0.0))

        clock_tick = AxisArray.LinearAxis(gain=0.05, offset=0.0)

        # Generate 4 chunks
        results = [producer(clock_tick) for _ in range(4)]
        agg = AxisArray.concatenate(*results, dim="time")

        # Should be continuous sine wave
        t = np.arange(200) / fs
        expected = np.sin(2 * np.pi * freq * t)
        np.testing.assert_allclose(agg.data, expected, rtol=1e-10)

    def test_sine_with_amplitude_and_phase(self):
        """Test sine wave with custom amplitude and phase."""
        fs = 1000.0
        freq = 5.0
        amp = 2.5
        phase = np.pi / 4
        producer = SinProducer(SinProducerSettings(fs=fs, n_time=100, freq=freq, amp=amp, phase=phase))

        clock_tick = AxisArray.LinearAxis(gain=0.1, offset=0.0)

        result = producer(clock_tick)

        t = np.arange(100) / fs
        expected = amp * np.sin(2 * np.pi * freq * t + phase)
        np.testing.assert_allclose(result.data, expected, rtol=1e-10)


class TestClockDrivenProducerWithExternalClock:
    """Tests simulating external clock patterns."""

    def test_external_clock_with_fixed_n_time(self):
        """Test external clock mode with fixed n_time."""
        producer = CounterProducer(CounterProducerSettings(fs=1000.0, n_time=100, mod=None))

        # Simulate external clock ticks
        timestamps = [1.0, 1.1, 1.2, 1.3, 1.4]
        clock_ticks = [AxisArray.LinearAxis(gain=0.1, offset=ts) for ts in timestamps]

        results = [producer(tick) for tick in clock_ticks]

        # Verify offsets match clock timestamps
        offsets = [r.axes["time"].offset for r in results]
        np.testing.assert_array_equal(offsets, timestamps)

        # Verify data continuity
        agg = AxisArray.concatenate(*results, dim="time")
        np.testing.assert_array_equal(agg.data, np.arange(500))

    def test_external_clock_variable_chunk_sizes(self):
        """Test external clock mode with variable chunk sizes."""
        producer = CounterProducer(CounterProducerSettings(fs=1000.0, n_time=None, mod=None))

        # Clock with varying rates
        clock_ticks = [
            AxisArray.LinearAxis(gain=0.1, offset=0.0),  # 100 samples
            AxisArray.LinearAxis(gain=0.05, offset=0.1),  # 50 samples
            AxisArray.LinearAxis(gain=0.2, offset=0.15),  # 200 samples
        ]

        results = [producer(tick) for tick in clock_ticks]

        assert results[0].data.shape == (100,)
        assert results[1].data.shape == (50,)
        assert results[2].data.shape == (200,)

        # Verify data continuity
        agg = AxisArray.concatenate(*results, dim="time")
        np.testing.assert_array_equal(agg.data, np.arange(350))


class TestClockDrivenStateManagement:
    """Tests for state management behavior."""

    def test_state_counter_increments(self):
        """Test that internal counter increments correctly."""
        producer = CounterProducer(CounterProducerSettings(fs=100.0, n_time=10, mod=None))

        clock_tick = AxisArray.LinearAxis(gain=0.1, offset=0.0)

        assert producer._state.counter == 0

        producer(clock_tick)
        assert producer._state.counter == 10

        producer(clock_tick)
        assert producer._state.counter == 20

    def test_fractional_samples_accumulate(self):
        """Test that fractional samples accumulate correctly."""
        producer = CounterProducer(CounterProducerSettings(fs=1000.0, n_time=None, mod=None))

        # Clock at 3 Hz -> 333.33... samples per tick
        clock_tick = AxisArray.LinearAxis(gain=1.0 / 3.0, offset=0.0)

        assert producer._state.fractional_samples == 0.0

        producer(clock_tick)
        # After first tick: 333.33... - 333 ≈ 0.33...
        assert 0.33 < producer._state.fractional_samples < 0.34

        producer(clock_tick)
        # After second tick: 333.33... + 0.33... - 333 ≈ 0.66...
        assert 0.66 < producer._state.fractional_samples < 0.67

    def test_hash_returns_constant(self):
        """Test that _hash_message returns constant (no state reset on clock changes)."""
        producer = CounterProducer(CounterProducerSettings(fs=100.0, n_time=10, mod=None))

        tick1 = AxisArray.LinearAxis(gain=0.1, offset=0.0)
        tick2 = AxisArray.LinearAxis(gain=0.2, offset=1.0)
        tick3 = AxisArray.LinearAxis(gain=0.0, offset=5.0)

        # All should return same hash (0)
        assert producer._hash_message(tick1) == 0
        assert producer._hash_message(tick2) == 0
        assert producer._hash_message(tick3) == 0
