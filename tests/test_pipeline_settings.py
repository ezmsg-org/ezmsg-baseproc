"""Unit tests for ezmsg.baseproc.util.pipeline_settings."""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import ezmsg.core as ez
import numpy as np
import pytest

from ezmsg.baseproc import (
    PipelineSettingsEvent,
    PipelineSettingsEventType,
    PipelineSettingsProducer,
    PipelineSettingsProducerSettings,
    flatten_component_settings,
    flatten_ez_settings,
    sanitize_settings_column_name,
    sanitize_settings_value,
)

# ---------------------------------------------------------------------------
# Helper tests (ported from ezmsg-nwb's test_writer.py)
# ---------------------------------------------------------------------------


@dataclass
class _Endpoint:
    host: str = "127.0.0.1"
    port: int = 5000


class _NestedSettings(ez.Settings):
    endpoint: _Endpoint = field(default_factory=_Endpoint)
    route_map: dict[str, str] = field(default_factory=lambda: {"open": "A", "closed": "B"})
    bands: list[tuple[float, float]] = field(default_factory=lambda: [(70.0, 200.0)])


def test_flatten_component_settings_handles_dataclasses_and_mappings():
    flat = flatten_component_settings("PIPELINE.UNIT", _NestedSettings())

    assert flat["PIPELINE.UNIT._NestedSettings.endpoint.host"] == "127.0.0.1"
    assert flat["PIPELINE.UNIT._NestedSettings.endpoint.port"] == 5000
    assert flat["PIPELINE.UNIT._NestedSettings.route_map.open"] == "A"
    assert flat["PIPELINE.UNIT._NestedSettings.route_map.closed"] == "B"
    np.testing.assert_array_equal(flat["PIPELINE.UNIT._NestedSettings.bands"], np.array([[70.0, 200.0]]))


class _Mode(Enum):
    TRAIN = "train"


class _EdgeCaseSettings(ez.Settings):
    mode: _Mode = _Mode.TRAIN
    config_path: Path = Path("/tmp/model.pt")
    sample_count: np.int64 = np.int64(7)
    labels: set[str] = field(default_factory=lambda: {"b", "a"})


def test_flatten_component_settings_sanitizes_edge_types():
    flat = flatten_component_settings("PIPELINE.EDGE", _EdgeCaseSettings())

    assert flat["PIPELINE.EDGE._EdgeCaseSettings.mode"] == "train"
    assert flat["PIPELINE.EDGE._EdgeCaseSettings.config_path"] == str(Path("/tmp/model.pt"))
    assert flat["PIPELINE.EDGE._EdgeCaseSettings.sample_count"] == 7
    np.testing.assert_array_equal(flat["PIPELINE.EDGE._EdgeCaseSettings.labels"], np.array(["a", "b"]))


def test_sanitize_settings_value_none_becomes_string():
    assert sanitize_settings_value(None) == "None"


def test_sanitize_settings_column_name_strips_unsafe_chars():
    assert sanitize_settings_column_name("foo/bar baz") == "foo.bar.baz"
    assert sanitize_settings_column_name("123foo") == "setting_123foo"
    assert sanitize_settings_column_name("___") == "setting"


def test_flatten_ez_settings_uses_class_name_as_default_prefix():
    flat = flatten_ez_settings(_EdgeCaseSettings())
    assert all(k.startswith("_EdgeCaseSettings.") for k in flat)


# ---------------------------------------------------------------------------
# PipelineSettingsEvent
# ---------------------------------------------------------------------------


_UNSET = object()


def _make_event(structured=_UNSET, repr_value=_UNSET, **overrides):
    defaults = dict(
        seq=42,
        timestamp=1234567.5,
        component_address="MY.UNIT",
        event_type=PipelineSettingsEventType.UPDATED,
        repr_value=repr_value if repr_value is not _UNSET else {"foo": 1, "bar": "x"},
        structured_value=structured if structured is not _UNSET else {"foo": 1, "bar": "x"},
    )
    defaults.update(overrides)
    return PipelineSettingsEvent(**defaults)


def test_event_flatten_for_table_returns_data_with_self_describing_json():
    ev = _make_event()
    out = ev.flatten_for_table()
    assert set(out.keys()) == {"data"}
    payload = json.loads(out["data"])
    assert payload == {
        "component": "MY.UNIT",
        "event_type": "UPDATED",
        "seq": 42,
        "settings": {"foo": 1, "bar": "x"},
    }


def test_event_flatten_for_table_falls_back_to_repr_value_when_no_structured():
    ev = _make_event(structured=None, repr_value="<NotADataclass(...)>")
    out = ev.flatten_for_table()
    payload = json.loads(out["data"])
    assert payload["settings"] == "<NotADataclass(...)>"


def test_event_flatten_for_table_uses_repr_dict_when_no_structured():
    ev = _make_event(structured=None, repr_value={"only": "repr"})
    out = ev.flatten_for_table()
    payload = json.loads(out["data"])
    assert payload["settings"] == {"only": "repr"}


def test_event_default_table_name():
    ev = _make_event()
    assert ev.table_name == "settings_annotations"


# ---------------------------------------------------------------------------
# Producer reset failure path (no graph server running)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_producer_handles_missing_graph_server_gracefully():
    """When no GraphServer is reachable, reset should warn and leave the queue
    empty rather than crash."""
    producer = PipelineSettingsProducer(PipelineSettingsProducerSettings())
    # GraphContext.__aenter__ tries to connect; with auto_start=False and no
    # server up, it raises. The producer catches and logs a warning.
    await producer._areset_state()
    assert producer._state.queue is None or producer._state.queue.empty()
    assert producer._state.initialized is False
    # Teardown is idempotent; should not raise.
    await producer._teardown()


def test_producer_sync_reset_raises_not_implemented():
    """The sync reset path is intentionally a guard; producer needs an event loop."""
    producer = PipelineSettingsProducer(PipelineSettingsProducerSettings())
    with pytest.raises(NotImplementedError):
        producer._reset_state()
