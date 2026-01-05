"""Clock-driven producer base classes for generating data synchronized to clock ticks."""

import typing
from abc import abstractmethod

import ezmsg.core as ez
from ezmsg.util.messages.axisarray import AxisArray, LinearAxis

from .protocols import StateType, processor_state
from .stateful import BaseStatefulProcessor


class ClockDrivenSettings(ez.Settings):
    """
    Base settings for clock-driven producers.

    Subclass this to add your own settings while inheriting fs and n_time.

    Example::

        class SinGeneratorSettings(ClockDrivenSettings):
            freq: float = 1.0
            amp: float = 1.0
    """

    fs: float
    """Output sampling rate in Hz."""

    n_time: int | None = None
    """
    Samples per block.
    - If specified: fixed chunk size (clock gain is ignored for determining chunk size)
    - If None: derived from clock gain (fs * clock.gain), with fractional sample tracking
    """


# Type variable for settings that extend ClockDrivenSettings
ClockDrivenSettingsType = typing.TypeVar("ClockDrivenSettingsType", bound=ClockDrivenSettings)


@processor_state
class ClockDrivenState:
    """
    Internal state for clock-driven producers.

    Tracks sample counting and fractional sample accumulation.
    Subclasses should extend this if they need additional state.
    """

    counter: int = 0
    """Current sample counter (total samples produced)."""

    fractional_samples: float = 0.0
    """Accumulated fractional samples for variable chunk mode."""


class BaseClockDrivenProducer(
    BaseStatefulProcessor[ClockDrivenSettingsType, AxisArray.LinearAxis, AxisArray, StateType],
    typing.Generic[ClockDrivenSettingsType, StateType],
):
    """
    Base class for clock-driven data producers.

    Accepts clock ticks (LinearAxis) as input and produces AxisArray output.
    Handles all the timing/counter logic internally, so subclasses only need
    to implement the data generation logic.

    This eliminates the need for the Clock → Counter → Generator pattern
    by combining the Counter functionality into the generator base class.

    Subclasses must implement:
        - ``_reset_state(time_axis)``: Initialize any state needed for production
        - ``_produce(n_samples, time_axis)``: Generate the actual output data

    Example::

        @processor_state
        class SinState(ClockDrivenState):
            ang_freq: float = 0.0

        class SinProducer(BaseClockDrivenProducer[SinSettings, SinState]):
            def _reset_state(self, time_axis: AxisArray.TimeAxis) -> None:
                self._state.ang_freq = 2 * np.pi * self.settings.fs

            def _produce(self, n_samples: int, time_axis: AxisArray.TimeAxis) -> AxisArray:
                t = (np.arange(n_samples) + self._state.counter) * time_axis.gain
                data = np.sin(self._state.ang_freq * t)
                return AxisArray(data=data, dims=["time"], axes={"time": time_axis})
    """

    def _hash_message(self, message: AxisArray.LinearAxis) -> int:
        # Return constant hash - state should not reset based on clock rate changes.
        # The producer maintains continuity regardless of clock rate changes.
        return 0

    def _compute_samples_and_offset(self, clock_tick: AxisArray.LinearAxis) -> tuple[int, float] | None:
        """
        Compute number of samples and time offset from a clock tick.

        Returns:
            Tuple of (n_samples, offset) or None if no samples to produce yet.

        Raises:
            ValueError: If clock gain is 0 (AFAP mode) and n_time is not specified.
        """
        if self.settings.n_time is not None:
            # Fixed chunk size mode
            n_samples = self.settings.n_time
            if clock_tick.gain == 0.0:
                # AFAP mode - synthetic offset based on counter
                offset = self._state.counter / self.settings.fs
            else:
                # Use clock's timestamp
                offset = clock_tick.offset
        else:
            # Variable chunk size mode - derive from clock gain
            if clock_tick.gain == 0.0:
                raise ValueError("Cannot use clock with gain=0 (AFAP) without specifying n_time")

            # Calculate samples including fractional accumulation
            samples_float = self.settings.fs * clock_tick.gain + self._state.fractional_samples
            n_samples = int(samples_float + 1e-9)
            self._state.fractional_samples = samples_float - n_samples

            if n_samples == 0:
                return None

            offset = clock_tick.offset

        return n_samples, offset

    @abstractmethod
    def _reset_state(self, time_axis: LinearAxis) -> None:
        """
        Reset/initialize state for production.

        Called once before the first call to _produce, or when state needs resetting.
        Use this to pre-compute values, create templates, etc.

        Args:
            time_axis: TimeAxis with the output sampling rate (fs) and initial offset.
        """
        ...

    @abstractmethod
    def _produce(self, n_samples: int, time_axis: LinearAxis) -> AxisArray:
        """
        Generate output data for this chunk.

        Args:
            n_samples: Number of samples to generate.
            time_axis: TimeAxis with correct offset and gain (1/fs) for this chunk.

        Returns:
            AxisArray containing the generated data. The time axis should use
            the provided time_axis or one derived from it.
        """
        ...

    def _process(self, clock_tick: LinearAxis) -> AxisArray | None:
        """
        Process a clock tick and produce output.

        Handles all the counter/timing logic internally, then calls _produce.
        """
        result = self._compute_samples_and_offset(clock_tick)
        if result is None:
            return None

        n_samples, offset = result
        time_axis = AxisArray.TimeAxis(fs=self.settings.fs, offset=offset)

        # Call subclass production method
        output = self._produce(n_samples, time_axis)

        # Update counter
        self._state.counter += n_samples

        return output
