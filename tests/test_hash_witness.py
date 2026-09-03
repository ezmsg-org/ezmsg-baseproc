"""The state-hash witness must be invisible: same answer, less work.

``_message_hash`` caches its result against a *witness* -- the objects it was
derived from -- and returns the cached value when none of them has changed. The
only thing that can go wrong is returning a stale hash, and the consequence is a
processor that silently keeps state belonging to a configuration that is gone.
So the property under test is not "the fast path is fast" but "the fast path is
indistinguishable from recomputing", checked against a randomised stream of every
mutation a real one can undergo.

Three bugs were found this way and each has a named test below: a witness blind
to ``exclude_dims``, a specialised validator that assumed the chunk axis stayed
linear, and a generic one that skipped an excluded dimension whose axis had
disappeared.
"""

from __future__ import annotations

import random
import typing

import numpy as np
import pytest
from ezmsg.util.messages.axisarray import AxisArray, CoordinateAxis

from ezmsg.baseproc.stateful import Stateful


class Probe(Stateful[dict]):
    """The base-class hash, with nothing else attached."""

    def _reset_state(self, message: typing.Any) -> None: ...

    def _process(self, message: typing.Any) -> typing.Any:
        return message

    def stateful_op(self, state: typing.Any, message: typing.Any) -> typing.Any:
        raise NotImplementedError


def recomputed(message: AxisArray, **kwargs: typing.Any) -> int:
    """The same hash with the witness disabled -- the reference answer."""
    probe = Probe()
    probe._hash_witness = None
    return probe._message_hash(message, **kwargs)


def msg(
    labels: list[str],
    *,
    fs: float = 100.0,
    key: str = "dev",
    n_chunk: int = 8,
    offset: float = 0.0,
    ch_axis: CoordinateAxis | None = None,
    chunk_dim: str | None = "time",
) -> AxisArray:
    return AxisArray(
        np.zeros((n_chunk, len(labels)), np.float32),
        dims=["time", "ch"],
        axes={
            "time": AxisArray.TimeAxis(fs=fs, offset=offset),
            "ch": ch_axis if ch_axis is not None else CoordinateAxis(data=np.array(labels), dims=["ch"]),
        },
        key=key,
        **({"chunk_dim": chunk_dim} if chunk_dim else {}),
    )


class TestTheFastPathAgreesWithRecomputing:
    def test_a_reused_axis_object_hits_and_agrees(self):
        """The template idiom: one axis object for the life of the stream."""
        probe = Probe()
        hoisted = CoordinateAxis(data=np.array(["a", "b"]), dims=["ch"])
        first = probe._message_hash(msg(["a", "b"], ch_axis=hoisted))
        probe._hash = first
        for step in range(1, 6):
            m = msg(["a", "b"], ch_axis=hoisted, offset=step * 0.1, n_chunk=8 + step)
            assert probe._message_hash(m) == recomputed(m) == first

    def test_a_rebuilt_axis_with_equal_content_still_agrees(self):
        """What every consumer sees on the far side of a process boundary:
        a new object each message, carrying the same values."""
        probe = Probe()
        first = probe._message_hash(msg(["a", "b"]))
        probe._hash = first
        for step in range(1, 6):
            m = msg(["a", "b"], offset=step * 0.1)
            assert probe._message_hash(m) == recomputed(m) == first

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda: msg(["a", "z"]), id="relabelled"),
            pytest.param(lambda: msg(["a", "b", "c"]), id="channel_count"),
            pytest.param(lambda: msg(["a", "b"], fs=200.0), id="sample_rate"),
            pytest.param(lambda: msg(["a", "b"], key="other"), id="key"),
        ],
    )
    def test_a_real_change_is_not_masked(self, mutate):
        probe = Probe()
        before = probe._message_hash(msg(["a", "b"]))
        probe._hash = before
        after = probe._message_hash(mutate())
        assert after == recomputed(mutate())
        assert after != before

    def test_withdrawing_chunk_dim_changes_nothing_here(self):
        """Undeclared falls back to ``STREAMING_DIMS``, which names the same
        dimension for a ``(time, ch)`` stream -- so the hash is unchanged, and
        the witness has to agree rather than assume a declaration change matters."""
        probe = Probe()
        before = probe._message_hash(msg(["a", "b"]))
        probe._hash = before
        undeclared = msg(["a", "b"], chunk_dim=None)
        assert probe._message_hash(undeclared) == recomputed(undeclared) == before

    def test_chunk_size_jitter_does_not_disturb_it(self):
        probe = Probe()
        hoisted = CoordinateAxis(data=np.array(["a", "b"]), dims=["ch"])
        first = probe._message_hash(msg(["a", "b"], ch_axis=hoisted, n_chunk=8))
        probe._hash = first
        assert probe._message_hash(msg(["a", "b"], ch_axis=hoisted, n_chunk=37)) == first


class TestTheBugsTheFuzzFound:
    def test_exclude_dims_is_part_of_the_witness(self):
        """A witness built for one exclusion set must not answer for another."""
        probe = Probe()
        hoisted = CoordinateAxis(data=np.array(["a", "b"]), dims=["ch"])
        m = msg(["a", "b"], ch_axis=hoisted)
        probe._hash = probe._message_hash(m)
        assert probe._message_hash(m, exclude_dims=("ch",)) == recomputed(m, exclude_dims=("ch",))

    def test_a_chunk_axis_that_stops_being_linear(self):
        """An irregular-rate stream swaps its TimeAxis for a CoordinateAxis. The
        specialised validator reads ``.gain`` directly and must not raise."""
        probe = Probe()
        hoisted = CoordinateAxis(data=np.array(["a", "b"]), dims=["ch"])
        probe._hash = probe._message_hash(msg(["a", "b"], ch_axis=hoisted))
        irregular = AxisArray(
            np.zeros((8, 2), np.float32),
            dims=["time", "ch"],
            axes={
                "time": CoordinateAxis(data=np.arange(8).astype(float), dims=["time"], unit="s"),
                "ch": hoisted,
            },
            key="dev",
            chunk_dim="time",
        )
        assert probe._message_hash(irregular) == recomputed(irregular)

    def test_an_excluded_dimension_losing_its_axis(self):
        """Absence drops a term from the hash, so it must compare unequal to a
        gain rather than be skipped."""
        probe = Probe()
        hoisted = CoordinateAxis(data=np.array(["a", "b"]), dims=["ch"])
        with_axis = AxisArray(
            np.zeros((8, 2, 2), np.float32),
            dims=["time", "ch", "feat"],
            axes={
                "time": AxisArray.TimeAxis(fs=100.0),
                "ch": hoisted,
                "feat": AxisArray.LinearAxis(gain=2.0, offset=0.0),
            },
            key="dev",
            chunk_dim="time",
        )
        without = AxisArray(
            np.zeros((8, 2, 2), np.float32),
            dims=["time", "ch", "feat"],
            axes={"time": AxisArray.TimeAxis(fs=100.0), "ch": hoisted},
            key="dev",
            chunk_dim="time",
        )
        kwargs = {"exclude_dims": ("feat",)}
        probe._hash = probe._message_hash(with_axis, **kwargs)
        assert probe._message_hash(without, **kwargs) == recomputed(without, **kwargs)


class TestTheWitnessIsDroppedWhenItMustBe:
    def test_restoring_state_drops_it(self):
        """``stateful_op`` hands in state built elsewhere; a witness describing
        the old state would answer with a hash for state that is gone."""
        probe = Probe()
        probe._message_hash(msg(["a", "b"]))
        assert probe._hash_witness is not None
        probe.state = {}
        assert probe._hash_witness is None

    def test_it_survives_nothing_it_should_not(self):
        probe = Probe()
        probe._message_hash(msg(["a", "b"]))
        probe._hash_witness = None
        m = msg(["a", "b"], offset=0.5)
        assert probe._message_hash(m) == recomputed(m)


DIMSETS = [
    (["time", "ch"], "time"),
    (["win", "time", "ch"], "win"),
    (["time", "ch"], None),
    (["ch", "time"], "time"),
    (["time", "ch", "feat"], "time"),
    (["ch", "feat", "time"], "time"),
]
CALL_KWARGS = [
    {},
    {"include_key": False},
    {"extra": (7, "a")},
    {"exclude_dims": ("ch",)},
    {"exclude_dims": ("feat",), "include_key": False},
]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fuzz_the_fast_path_never_disagrees(seed: int):
    """Randomised streams: every message's witnessed hash must equal its
    recomputed one, whatever the stream does between messages."""
    rng = random.Random(seed)
    labels_pool = [["a", "b", "c"], ["x", "y", "z"], ["a", "b", "c", "d"], ["a", "b"]]

    def build(dims, chunk, labels, fs, key, n_chunk, offset, coord_time):
        shape, axes = [], {}
        for dim in dims:
            if dim == "ch":
                shape.append(len(labels))
                axes["ch"] = CoordinateAxis(data=np.array(labels), dims=["ch"])
            elif dim in ("time", "win"):
                shape.append(n_chunk)
                axes[dim] = (
                    CoordinateAxis(data=np.arange(n_chunk).astype(float), dims=[dim], unit="s")
                    if coord_time
                    else AxisArray.TimeAxis(fs=fs, offset=offset)
                )
            else:
                shape.append(2)
                roll = rng.random()
                if roll < 0.4:
                    axes[dim] = CoordinateAxis(data=np.array([f"{dim}0", f"{dim}1"]), dims=[dim])
                elif roll < 0.7:
                    axes[dim] = AxisArray.LinearAxis(gain=rng.choice([1.0, 2.0]), offset=rng.choice([0.0, 5.0]))
        extra = {"chunk_dim": chunk} if chunk in dims else {}
        return AxisArray(np.zeros(shape, np.float32), dims=list(dims), axes=axes, key=key, **extra)

    checked = 0
    for _ in range(120):
        probe = Probe()
        dims, chunk = rng.choice(DIMSETS)
        kwargs = rng.choice(CALL_KWARGS)
        labels, fs, key = rng.choice(labels_pool), rng.choice([100.0, 200.0]), rng.choice(["dev", "dev2"])
        hoisted = CoordinateAxis(data=np.array(labels), dims=["ch"])
        for step in range(14):
            roll = rng.random()
            if roll < 0.15:
                labels = rng.choice(labels_pool)
                hoisted = CoordinateAxis(data=np.array(labels), dims=["ch"])
            elif roll < 0.25:
                fs = rng.choice([100.0, 200.0])
            elif roll < 0.32:
                key = rng.choice(["dev", "dev2"])
            elif roll < 0.38:
                dims, chunk = rng.choice(DIMSETS)
            elif roll < 0.44:
                kwargs = rng.choice(CALL_KWARGS)
            elif roll < 0.48:
                probe.state = {}  # a stateful_op restore mid-stream
            message = build(dims, chunk, labels, fs, key, rng.choice([8, 13, 21]), step * 0.1, rng.random() < 0.15)
            # Half the time the producer hands back the same axis object.
            if "ch" in message.axes and rng.random() < 0.5:
                if len(labels) == message.data.shape[message.dims.index("ch")]:
                    message.axes["ch"] = hoisted
            got = probe._message_hash(message, **kwargs)
            assert got == recomputed(
                message, **kwargs
            ), f"stale hash: dims={message.dims} chunk_dim={message.chunk_dim} kwargs={kwargs}"
            probe._hash = got
            checked += 1
    assert checked == 120 * 14
