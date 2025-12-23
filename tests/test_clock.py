"""Unit tests for ezmsg.baseproc.clock module."""

import math
import time

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray

from ezmsg.baseproc import ClockProducer, ClockSettings


@pytest.mark.parametrize("dispatch_rate", [math.inf, 1.0, 2.0, 5.0, 10.0, 20.0])
def test_clock_producer_sync(dispatch_rate: float):
    """Test synchronous ClockProducer via __call__."""
    run_time = 1.0
    n_target = 100 if math.isinf(dispatch_rate) else int(np.ceil(dispatch_rate * run_time))

    producer = ClockProducer(ClockSettings(dispatch_rate=dispatch_rate))

    results = []
    t_start = time.monotonic()
    while len(results) < n_target:
        results.append(producer())
    t_elapsed = time.monotonic() - t_start

    # All results should be LinearAxis
    assert all(isinstance(r, AxisArray.LinearAxis) for r in results)

    # Check gain values
    if math.isinf(dispatch_rate):
        assert all(r.gain == 0.0 for r in results)
    else:
        expected_gain = 1.0 / dispatch_rate
        assert all(abs(r.gain - expected_gain) < 1e-10 for r in results)

    # Offsets (timestamps) should be monotonically increasing
    offsets = [r.offset for r in results]
    assert all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1))

    # Check timing
    if math.isfinite(dispatch_rate):
        assert (run_time - 1 / dispatch_rate) < t_elapsed < (run_time + 0.2)
    else:
        # 100 usec per iteration is pretty generous for AFAP
        assert t_elapsed < (n_target * 1e-4)


@pytest.mark.parametrize("dispatch_rate", [math.inf, 2.0, 20.0])
@pytest.mark.asyncio
async def test_clock_producer_async(dispatch_rate: float):
    """Test asynchronous ClockProducer via __acall__."""
    run_time = 1.0
    n_target = 100 if math.isinf(dispatch_rate) else int(np.ceil(dispatch_rate * run_time))

    producer = ClockProducer(ClockSettings(dispatch_rate=dispatch_rate))

    results = []
    t_start = time.monotonic()
    while len(results) < n_target:
        results.append(await producer.__acall__())
    t_elapsed = time.monotonic() - t_start

    # All results should be LinearAxis
    assert all(isinstance(r, AxisArray.LinearAxis) for r in results)

    # Check gain values
    if math.isinf(dispatch_rate):
        assert all(r.gain == 0.0 for r in results)
    else:
        expected_gain = 1.0 / dispatch_rate
        assert all(abs(r.gain - expected_gain) < 1e-10 for r in results)

    # Offsets (timestamps) should be monotonically increasing
    offsets = [r.offset for r in results]
    assert all(offsets[i] <= offsets[i + 1] for i in range(len(offsets) - 1))

    # Check timing
    if math.isfinite(dispatch_rate):
        assert (run_time - 1.1 / dispatch_rate) < t_elapsed < (run_time + 0.1)
    else:
        # 100 usec per iteration is pretty generous for AFAP
        assert t_elapsed < (n_target * 1e-4)
