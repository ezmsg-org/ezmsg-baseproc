"""Tests for Unit-level close-on-recreate behavior in ezmsg-baseproc."""

from __future__ import annotations

import dataclasses

import pytest

from ezmsg.baseproc import (
    BaseConsumer,
    BaseConsumerUnit,
    BaseProducer,
    BaseProducerUnit,
    BaseTransformer,
    BaseTransformerUnit,
)
from ezmsg.baseproc.units import _close_previous


@dataclasses.dataclass
class _Settings:
    pass


class _MsgIn:
    pass


class _MsgOut:
    pass


# --- Helper-level tests ---------------------------------------------------


def test_close_previous_handles_none() -> None:
    _close_previous(None)


def test_close_previous_handles_object_without_close() -> None:
    class NoClose:
        pass

    _close_previous(NoClose())


def test_close_previous_swallows_exceptions() -> None:
    class Raises:
        def close(self) -> None:
            raise RuntimeError("boom")

    # Must not propagate.
    _close_previous(Raises())


def test_close_previous_calls_close() -> None:
    closed = []

    class Tracker:
        def close(self) -> None:
            closed.append(True)

    _close_previous(Tracker())
    assert closed == [True]


# --- Unit-level tests -----------------------------------------------------


class _TrackProducer(BaseProducer[_Settings, _MsgOut]):
    instances: list["_TrackProducer"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closed = False
        _TrackProducer.instances.append(self)

    async def _produce(self) -> _MsgOut:
        return _MsgOut()

    def close(self) -> None:
        self.closed = True


class _TrackProducerUnit(BaseProducerUnit[_Settings, _MsgOut, _TrackProducer]):
    SETTINGS = _Settings


class _TrackTransformer(BaseTransformer[_Settings, _MsgIn, _MsgOut]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closed = False

    def _process(self, message: _MsgIn) -> _MsgOut:
        return _MsgOut()

    def close(self) -> None:
        self.closed = True


class _TrackTransformerUnit(BaseTransformerUnit[_Settings, _MsgIn, _MsgOut, _TrackTransformer]):
    SETTINGS = _Settings


class _TrackConsumer(BaseConsumer[_Settings, _MsgIn]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.closed = False

    def _process(self, message: _MsgIn) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _TrackConsumerUnit(BaseConsumerUnit[_Settings, _MsgIn, _TrackConsumer]):
    SETTINGS = _Settings


@pytest.fixture(autouse=True)
def _clear_producer_instances():
    _TrackProducer.instances.clear()
    yield
    _TrackProducer.instances.clear()


def test_producer_unit_first_create_does_not_close() -> None:
    unit = _TrackProducerUnit(_Settings())
    unit.create_producer()
    assert unit.producer.closed is False


def test_producer_unit_recreate_closes_previous() -> None:
    unit = _TrackProducerUnit(_Settings())
    unit.create_producer()
    first = unit.producer
    unit.create_producer()
    second = unit.producer

    assert first is not second
    assert first.closed, "previous producer should have been closed"
    assert second.closed is False


def test_transformer_unit_recreate_closes_previous() -> None:
    unit = _TrackTransformerUnit(_Settings())
    unit.create_processor()
    first = unit.processor
    unit.create_processor()
    second = unit.processor

    assert first is not second
    assert first.closed
    assert second.closed is False


def test_consumer_unit_recreate_closes_previous() -> None:
    unit = _TrackConsumerUnit(_Settings())
    unit.create_processor()
    first = unit.processor
    unit.create_processor()
    second = unit.processor

    assert first is not second
    assert first.closed
    assert second.closed is False


def test_default_close_on_base_producer_is_noop() -> None:
    class Bare(BaseProducer[_Settings, _MsgOut]):
        async def _produce(self) -> _MsgOut:
            return _MsgOut()

    Bare(_Settings()).close()  # must not raise


def test_default_close_on_base_processor_is_noop() -> None:
    class Bare(BaseTransformer[_Settings, _MsgIn, _MsgOut]):
        def _process(self, message: _MsgIn) -> _MsgOut:
            return _MsgOut()

    Bare(_Settings()).close()  # must not raise
