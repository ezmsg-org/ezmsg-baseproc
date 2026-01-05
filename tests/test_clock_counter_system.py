"""Integration tests for Clock and Counter ezmsg systems."""

import math
import os
from dataclasses import field

import ezmsg.core as ez
import numpy as np
import pytest
from ezmsg.util.messagecodec import message_log
from ezmsg.util.messagelogger import MessageLogger, MessageLoggerSettings
from ezmsg.util.messages.axisarray import AxisArray
from ezmsg.util.terminate import TerminateOnTotal, TerminateOnTotalSettings

from ezmsg.baseproc import (
    Clock,
    ClockSettings,
    Counter,
    CounterSettings,
)
from tests.helpers.util import get_test_fn


class ClockTestSystemSettings(ez.Settings):
    clock_settings: ClockSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings = field(default_factory=TerminateOnTotalSettings)


class ClockTestSystem(ez.Collection):
    SETTINGS = ClockTestSystemSettings

    CLOCK = Clock()
    LOG = MessageLogger()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.CLOCK.apply_settings(self.SETTINGS.clock_settings)
        self.LOG.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.CLOCK.OUTPUT_SIGNAL, self.LOG.INPUT_MESSAGE),
            (self.LOG.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE),
        )


@pytest.mark.parametrize("dispatch_rate", [math.inf, 2.0, 20.0])
def test_clock_system(
    dispatch_rate: float,
    test_name: str | None = None,
):
    run_time = 1.0
    n_target = 100 if math.isinf(dispatch_rate) else int(np.ceil(dispatch_rate * run_time))
    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)
    settings = ClockTestSystemSettings(
        clock_settings=ClockSettings(dispatch_rate=dispatch_rate),
        log_settings=MessageLoggerSettings(output=test_filename),
        term_settings=TerminateOnTotalSettings(total=n_target),
    )
    system = ClockTestSystem(settings)
    ez.run(SYSTEM=system)

    # Collect result
    messages = list(message_log(test_filename))
    os.remove(test_filename)

    # Clock produces LinearAxis with gain and offset
    assert all(isinstance(m, AxisArray.LinearAxis) for m in messages)
    assert len(messages) >= n_target


class CounterTestSystemSettings(ez.Settings):
    clock_settings: ClockSettings
    counter_settings: CounterSettings
    log_settings: MessageLoggerSettings
    term_settings: TerminateOnTotalSettings = field(default_factory=TerminateOnTotalSettings)


class CounterTestSystem(ez.Collection):
    """Counter must be driven by Clock in the new architecture."""

    SETTINGS = CounterTestSystemSettings

    CLOCK = Clock()
    COUNTER = Counter()
    LOG = MessageLogger()
    TERM = TerminateOnTotal()

    def configure(self) -> None:
        self.CLOCK.apply_settings(self.SETTINGS.clock_settings)
        self.COUNTER.apply_settings(self.SETTINGS.counter_settings)
        self.LOG.apply_settings(self.SETTINGS.log_settings)
        self.TERM.apply_settings(self.SETTINGS.term_settings)

    def network(self) -> ez.NetworkDefinition:
        return (
            (self.CLOCK.OUTPUT_SIGNAL, self.COUNTER.INPUT_CLOCK),
            (self.COUNTER.OUTPUT_SIGNAL, self.LOG.INPUT_MESSAGE),
            (self.LOG.OUTPUT_MESSAGE, self.TERM.INPUT_MESSAGE),
        )


@pytest.mark.parametrize(
    "n_time, fs, dispatch_rate, mod",
    [
        (1, 10.0, math.inf, None),  # AFAP mode
        (20, 1000.0, 50.0, None),  # Realtime mode (50 Hz dispatch = 20 samples/tick @ 1000 Hz)
        (1, 1000.0, 100.0, 2**3),  # 100 Hz dispatch with mod
        (10, 10.0, 10.0, 2**3),  # 10 Hz dispatch with mod
    ],
)
def test_counter_system(
    n_time: int,
    fs: float,
    dispatch_rate: float,
    mod: int | None,
    test_name: str | None = None,
):
    target_dur = 2.6  # 2.6 seconds per test
    if math.isinf(dispatch_rate):
        # AFAP mode - runs as fast as possible
        target_messages = 100  # Fixed target for AFAP
    else:
        target_messages = int(target_dur * dispatch_rate)

    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)
    settings = CounterTestSystemSettings(
        clock_settings=ClockSettings(dispatch_rate=dispatch_rate),
        counter_settings=CounterSettings(
            n_time=n_time,
            fs=fs,
            mod=mod,
        ),
        log_settings=MessageLoggerSettings(
            output=test_filename,
        ),
        term_settings=TerminateOnTotalSettings(
            total=target_messages,
        ),
    )
    system = CounterTestSystem(settings)
    ez.run(SYSTEM=system)

    # Collect result
    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    if math.isinf(dispatch_rate):
        # The number of messages depends on how fast the computer is
        target_messages = len(messages)
    # This should be an equivalence assertion (==) but the use of TerminateOnTotal does
    #  not guarantee that MessageLogger will exit before an additional message is received.
    #  Let's just clip the last message if we exceed the target messages.
    if len(messages) > target_messages:
        messages = messages[:target_messages]
    assert len(messages) >= target_messages

    # Just do one quick data check (Counter now outputs 1D array)
    agg = AxisArray.concatenate(*messages, dim="time")
    target_samples = n_time * target_messages
    expected_data = np.arange(target_samples)
    if mod is not None:
        expected_data = expected_data % mod
    assert np.array_equal(agg.data, expected_data)


@pytest.mark.parametrize(
    "clock_rate, fs, n_time",
    [
        (10.0, 1000.0, 100),  # 10 Hz clock, fs=1000, n_time=100 (fixed)
        (20.0, 500.0, None),  # 20 Hz clock, fs=500, n_time derived (25 samples per tick)
        (5.0, 1000.0, None),  # 5 Hz clock, fs=1000, n_time derived (200 samples per tick)
    ],
)
def test_counter_with_external_clock(
    clock_rate: float,
    fs: float,
    n_time: int | None,
    test_name: str | None = None,
):
    """Test Counter driven by external Clock (now the standard pattern)."""
    target_messages = 20
    test_filename = get_test_fn(test_name)
    ez.logger.info(test_filename)

    # This now uses the same CounterTestSystem since all counters need clocks
    settings = CounterTestSystemSettings(
        clock_settings=ClockSettings(dispatch_rate=clock_rate),
        counter_settings=CounterSettings(
            fs=fs,
            n_time=n_time,
        ),
        log_settings=MessageLoggerSettings(output=test_filename),
        term_settings=TerminateOnTotalSettings(total=target_messages),
    )
    system = CounterTestSystem(settings)
    ez.run(SYSTEM=system)

    # Collect result
    messages: list[AxisArray] = [_ for _ in message_log(test_filename)]
    os.remove(test_filename)

    assert len(messages) >= target_messages

    # Verify each message has correct sample rate (gain = 1/fs)
    for msg in messages:
        assert msg.axes["time"].gain == 1.0 / fs

    # Verify data continuity
    messages = messages[:target_messages]  # Trim to target
    agg = AxisArray.concatenate(*messages, dim="time")

    # Expected samples per tick
    if n_time is not None:
        expected_samples_per_tick = n_time
    else:
        expected_samples_per_tick = int(fs / clock_rate)

    expected_total = expected_samples_per_tick * target_messages
    # Allow for fractional sample accumulation variance
    assert abs(len(agg.data) - expected_total) <= target_messages

    # Counter values should be sequential (0, 1, 2, ...)
    expected_data = np.arange(len(agg.data))
    assert np.array_equal(agg.data, expected_data)
