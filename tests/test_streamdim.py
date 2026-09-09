"""The three rules for deciding which dimension a processor operates on.

``dims[0]`` is not "the streaming axis" and ``dims[-1]`` is not "the channel
axis" -- both are positions, and both break under transpose and downstream of a
windowing stage. ``AxisArray.stream_dim`` is the producer's declaration of which
dimension messages accumulate along; these helpers turn it into the axis a given
kind of processor should use.

The fixtures are deliberately transposed or windowed, because that is the only
place the positional guess and the declaration disagree.
"""

import numpy as np
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.baseproc import (
    resolve_configured_stream_dim,
    resolve_feature_dim,
    resolve_stream_dim,
    resolve_transform_dim,
)

FS = 100.0


def _ch_axis(n):
    return CoordinateAxis(data=np.array([f"ch{i}" for i in range(n)]), dims=["ch"])


def transposed(n_ch=3, n_time=8, stream_dim="time"):
    """``(ch, time)``: the first dim is static, the accumulating one is second."""
    kwargs = {"stream_dim": stream_dim} if stream_dim else {}
    return AxisArray(
        np.arange(n_ch * n_time, dtype=float).reshape(n_ch, n_time),
        dims=["ch", "time"],
        axes={"ch": _ch_axis(n_ch), "time": AxisArray.TimeAxis(fs=FS)},
        key="dev",
        **kwargs,
    )


def windowed(n_win=4, n_time=8, n_ch=3):
    """``(win, time, ch)``: ``win`` accumulates, ``time`` is within-window."""
    return AxisArray(
        np.zeros((n_win, n_time, n_ch), dtype=float),
        dims=["win", "time", "ch"],
        axes={
            "win": AxisArray.TimeAxis(fs=FS / n_time),
            "time": AxisArray.TimeAxis(fs=FS),
            "ch": _ch_axis(n_ch),
        },
        key="dev",
        stream_dim="win",
    )


def raw(n_time=8, n_ch=3, stream_dim="time"):
    kwargs = {"stream_dim": stream_dim} if stream_dim else {}
    return AxisArray(
        np.zeros((n_time, n_ch), dtype=float),
        dims=["time", "ch"],
        axes={"time": AxisArray.TimeAxis(fs=FS), "ch": _ch_axis(n_ch)},
        key="dev",
        **kwargs,
    )


class TestResolveStreamDim:
    def test_the_declaration_wins_over_position(self):
        msg = transposed()
        assert msg.dims[0] != msg.stream_dim, "the fixture must distinguish the two"
        assert resolve_stream_dim(msg) == "time"

    def test_it_follows_a_windowing_stage_onto_win(self):
        assert resolve_stream_dim(windowed()) == "win"

    def test_an_undeclared_stream_dim_falls_back_to_streaming_dims(self):
        assert resolve_stream_dim(transposed(stream_dim=None)) == "time"

    def test_the_streaming_dims_fallback_is_configurable(self):
        msg = windowed()
        object.__setattr__(msg, "stream_dim", None)
        assert resolve_stream_dim(msg, ("win",)) == "win"

    def test_dims_zero_is_the_last_resort_only(self):
        """Nothing declared and nothing recognised: the position is all there is."""
        msg = AxisArray(np.zeros((4, 2)), dims=["a", "b"], key="dev")
        assert resolve_stream_dim(msg) == "a"


class TestResolveFeatureDim:
    def test_it_skips_the_stream_dim_on_a_transposed_stream(self):
        """``dims[-1]`` here is ``time``. A slicer defaulting to it would drop
        samples, and an affine transform would matmul across time."""
        msg = transposed()
        assert msg.dims[-1] == msg.stream_dim, "the fixture must make the naive guess wrong"
        assert resolve_feature_dim(msg) == "ch"

    def test_it_is_unchanged_on_a_conventional_stream(self):
        assert resolve_feature_dim(raw()) == "ch"

    def test_position_zero_skips_the_stream_dim_too(self):
        """RangedAggregate's case: ``dims[0]`` is usually the stream dim, which is
        the worst possible default for an axis that must carry band values."""
        assert resolve_feature_dim(windowed(), 0) == "time"

    def test_a_stream_only_message_falls_back_rather_than_raising(self):
        msg = AxisArray(np.zeros(8), dims=["time"], axes={"time": AxisArray.TimeAxis(fs=FS)}, stream_dim="time")
        assert resolve_feature_dim(msg) == "time"


class TestResolveTransformDim:
    def test_windowed_input_transforms_within_the_window(self):
        """``win`` accumulates, but each window's spectrum is over ``time``."""
        assert resolve_transform_dim(windowed()) == "time"

    def test_raw_input_falls_through_to_the_stream_dim(self):
        """``ch`` carries a CoordinateAxis, not a LinearAxis, so it is not a
        candidate and the rule lands back on ``time``."""
        assert resolve_transform_dim(raw()) == "time"

    def test_it_holds_under_transposition(self):
        assert resolve_transform_dim(transposed()) == "time"


class TestResolveConfiguredStreamDim:
    def test_an_explicit_axis_still_wins(self):
        """The escape hatch stays open: an explicit axis is an instruction."""

        class Proc:
            STREAMING_DIMS = ("time",)

        assert resolve_configured_stream_dim(Proc(), windowed(), "time") == "time"

    def test_a_disagreement_warns_once(self, caplog):
        class Proc:
            STREAMING_DIMS = ("time",)

        proc = Proc()
        with caplog.at_level("WARNING"):
            for _ in range(3):
                resolve_configured_stream_dim(proc, windowed(), "time")
        assert sum("stream_dim" in r.message for r in caplog.records) == 1

    def test_agreement_is_silent(self, caplog):
        class Proc:
            STREAMING_DIMS = ("time",)

        with caplog.at_level("WARNING"):
            resolve_configured_stream_dim(Proc(), raw(), "time")
        assert not caplog.records

    def test_a_mere_guess_never_warns(self, caplog):
        """Warning against STREAMING_DIMS rather than a declaration would fire on
        every correctly-configured windowed pipeline whose producer is silent."""

        class Proc:
            STREAMING_DIMS = ("time",)

        msg = windowed()
        object.__setattr__(msg, "stream_dim", None)
        with caplog.at_level("WARNING"):
            resolve_configured_stream_dim(Proc(), msg, "win")
        assert not caplog.records
