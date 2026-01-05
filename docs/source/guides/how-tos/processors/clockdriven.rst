How to implement a clock-driven producer?
#########################################

Clock-driven producers generate data synchronized to clock ticks. They are useful
for signal generators, simulators, and other components that need to produce
timed data streams.

The ``BaseClockDrivenProducer`` base class simplifies this pattern by handling
all the timing and sample counting logic internally. You only need to implement
the data generation.

When to use BaseClockDrivenProducer
===================================

Use ``BaseClockDrivenProducer`` when you need to:

- Generate synthetic signals (sine waves, noise, test patterns)
- Simulate sensor data at a specific sample rate
- Produce timed data streams driven by a ``Clock``

This base class eliminates the need for the ``Clock → Counter → Generator``
pattern by combining the counter functionality into the generator.

Basic Structure
===============

A clock-driven producer consists of three parts:

1. **Settings** - Extends ``ClockDrivenSettings`` (which provides ``fs`` and ``n_time``)
2. **State** - Extends ``ClockDrivenState`` (which provides ``counter`` and ``fractional_samples``)
3. **Producer** - Extends ``BaseClockDrivenProducer`` and implements ``_reset_state`` and ``_produce``

Example: Sine Wave Generator
============================

Here's a complete example of a sine wave generator:

.. code-block:: python

    import numpy as np
    from ezmsg.util.messages.axisarray import AxisArray, LinearAxis

    from ezmsg.baseproc import (
        BaseClockDrivenProducer,
        BaseClockDrivenProducerUnit,
        ClockDrivenSettings,
        ClockDrivenState,
        processor_state,
    )


    class SinGeneratorSettings(ClockDrivenSettings):
        """
        Settings for the sine wave generator.

        Inherits from ClockDrivenSettings which provides:
        - fs: Output sampling rate in Hz
        - n_time: Samples per block (optional, derived from clock if None)
        """

        freq: float = 1.0
        """Frequency of the sine wave in Hz."""

        amp: float = 1.0
        """Amplitude of the sine wave."""

        phase: float = 0.0
        """Initial phase in radians."""


    @processor_state
    class SinGeneratorState(ClockDrivenState):
        """
        State for the sine wave generator.

        Inherits from ClockDrivenState which provides:
        - counter: Current sample counter (total samples produced)
        - fractional_samples: For accumulating sub-sample timing
        """

        ang_freq: float = 0.0
        """Pre-computed angular frequency (2 * pi * freq)."""


    class SinGenerator(
        BaseClockDrivenProducer[SinGeneratorSettings, SinGeneratorState]
    ):
        """
        Generates sine wave data synchronized to clock ticks.
        """

        def _reset_state(self, time_axis: LinearAxis) -> None:
            """
            Initialize state. Called once before first production.

            Use this to pre-compute values that don't change between chunks.
            """
            self._state.ang_freq = 2 * np.pi * self.settings.freq

        def _produce(self, n_samples: int, time_axis: LinearAxis) -> AxisArray:
            """
            Generate sine wave data for this chunk.

            Args:
                n_samples: Number of samples to generate
                time_axis: LinearAxis with correct offset and gain (1/fs)

            Returns:
                AxisArray containing the sine wave data
            """
            # Calculate time values using the internal counter
            t = (np.arange(n_samples) + self._state.counter) * time_axis.gain

            # Generate sine wave
            data = self.settings.amp * np.sin(
                self._state.ang_freq * t + self.settings.phase
            )

            return AxisArray(
                data=data,
                dims=["time"],
                axes={"time": time_axis},
            )


    class SinGeneratorUnit(
        BaseClockDrivenProducerUnit[SinGeneratorSettings, SinGenerator]
    ):
        """
        ezmsg Unit wrapper for SinGenerator.

        Receives clock ticks on INPUT_CLOCK and outputs AxisArray on OUTPUT_SIGNAL.
        """

        SETTINGS = SinGeneratorSettings


Key Points
==========

**Settings inheritance**: Your settings class should extend ``ClockDrivenSettings``,
which provides:

- ``fs``: The output sampling rate in Hz
- ``n_time``: Optional fixed chunk size. If ``None``, chunk size is derived from
  the clock's gain (``fs * clock.gain``)

**State inheritance**: Your state class should extend ``ClockDrivenState``,
which provides:

- ``counter``: Tracks total samples produced (use this for continuous signals)
- ``fractional_samples``: Accumulates sub-sample timing for accurate chunk sizes

**The _produce method**: This is where you generate data. You receive:

- ``n_samples``: How many samples to generate this chunk
- ``time_axis``: A ``LinearAxis`` with the correct ``offset`` and ``gain`` (1/fs)

The base class automatically:

- Computes ``n_samples`` from clock timing or settings
- Manages the sample counter (incremented after ``_produce`` returns)
- Handles fractional sample accumulation for non-integer chunk sizes
- Supports both fixed ``n_time`` and variable chunk modes

Using Standalone (Outside ezmsg)
================================

Clock-driven producers can be used standalone for testing or offline processing:

.. code-block:: python

    from ezmsg.util.messages.axisarray import AxisArray

    # Create the producer
    producer = SinGenerator(SinGeneratorSettings(
        fs=1000.0,      # 1000 Hz sample rate
        n_time=100,     # 100 samples per chunk
        freq=10.0,      # 10 Hz sine wave
        amp=1.0,
    ))

    # Simulate clock ticks (LinearAxis with gain=1/dispatch_rate, offset=timestamp)
    clock_tick = AxisArray.LinearAxis(gain=0.1, offset=0.0)  # 10 Hz dispatch

    # Generate data
    result = producer(clock_tick)
    print(f"Shape: {result.data.shape}")  # (100,)
    print(f"Sample rate: {1/result.axes['time'].gain} Hz")  # 1000.0 Hz


Using with ezmsg
================

In an ezmsg pipeline, connect a ``Clock`` to your generator's ``INPUT_CLOCK``:

.. code-block:: python

    import ezmsg.core as ez
    from ezmsg.baseproc import Clock, ClockSettings


    class SinPipeline(ez.Collection):
        SETTINGS = SinGeneratorSettings

        CLOCK = Clock()
        GENERATOR = SinGeneratorUnit()

        def configure(self) -> None:
            self.CLOCK.apply_settings(ClockSettings(dispatch_rate=10.0))
            self.GENERATOR.apply_settings(self.SETTINGS)

        def network(self) -> ez.NetworkDefinition:
            return (
                (self.CLOCK.OUTPUT_SIGNAL, self.GENERATOR.INPUT_CLOCK),
            )


See Also
========

- :doc:`API Reference for clockdriven module <../../../api/generated/ezmsg.baseproc.clockdriven>`
- :doc:`stateful` - For general stateful processor patterns
- :doc:`unit` - For converting processors to ezmsg Units
