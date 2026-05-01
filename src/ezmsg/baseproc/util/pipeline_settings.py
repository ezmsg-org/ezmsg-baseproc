"""Pipeline-settings event producer and supporting helpers.

This module provides a generic way to capture every settings change happening
inside an ezmsg pipeline and emit each one as a self-contained message that
can travel from unit to unit on the graph.

Three layers:

1. Helpers — ``flatten_ez_settings`` / ``flatten_component_settings`` /
   ``sanitize_settings_value`` / ``sanitize_settings_column_name``. These
   project arbitrary settings objects (``ez.Settings``, ``dict``, dataclass,
   numpy array, enum, path, …) into either dotted-key/value pairs or
   NWB-compatible scalar/array values. Useful to any tabular sink, not just
   NWB.
2. Message — ``PipelineSettingsEvent`` carries one settings change. It mirrors
   the meaningful subset of ``ezmsg.core.SettingsChangedEvent`` (``seq``,
   ``timestamp``, ``component_address``, ``repr_value``, ``structured_value``)
   plus a ``table_name`` for tabular sinks and an ``event_type`` distinguishing
   the startup snapshot from later updates. ``flatten_for_table`` projects to
   the ``{"data": json_str}`` shape an ``NWBPointRow``-compatible sink expects.
3. Producer / Unit — ``PipelineSettingsProducer`` + ``PipelineSettingsUnit``
   open a ``GraphContext``, queue one ``INITIAL`` event per in-scope component
   (using the current settings snapshot), and stream subsequent ``UPDATED``
   events from the graph server's settings subscription.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import typing
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Optional

import ezmsg.core as ez
import numpy as np

from ..protocols import processor_state
from ..stateful import BaseStatefulProducer
from ..units import BaseProducerUnit

# ---------------------------------------------------------------------------
# Flatten / sanitize helpers (general-purpose; not NWB-specific)
# ---------------------------------------------------------------------------


def _flatten_value(value: Any, prefix: str) -> dict[str, Any]:
    """Flatten a settings value into dotted key/value pairs."""
    if isinstance(value, ez.Settings):
        return flatten_ez_settings(value, prefix)

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, sub_value in value.items():
            key_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_value(sub_value, key_prefix))
        return result

    return {prefix if prefix else "value": value}


def flatten_ez_settings(settings: ez.Settings, prefix: str = "") -> dict[str, Any]:
    """Flatten an ezmsg settings object into dotted key/value pairs."""
    settings_prefix = settings.__class__.__name__ if not prefix else prefix
    result: dict[str, Any] = {}
    for name, value in settings.__dict__.items():
        result.update(_flatten_value(value, f"{settings_prefix}.{name}"))
    return result


def sanitize_settings_column_name(name: str) -> str:
    """Convert a settings field path into a tabular-safe column name."""
    sanitized = re.sub(r"[^0-9A-Za-z_]+", ".", name).strip("_")
    if not sanitized:
        sanitized = "setting"
    if sanitized[0].isdigit():
        sanitized = f"setting_{sanitized}"
    return sanitized


def _sanitize_sequence_value(value: Sequence[Any]) -> Any:
    """Sanitize a sequence while preserving array-like values when possible."""
    sanitized = [sanitize_settings_value(item) for item in value]

    try:
        array_value = np.asarray(sanitized)
    except Exception:
        return json.dumps(sanitized, default=str)

    if array_value.dtype != object:
        return array_value

    if all(isinstance(item, str) for item in sanitized):
        return np.asarray(sanitized, dtype=str)

    if all(isinstance(item, bytes) for item in sanitized):
        return np.asarray(sanitized, dtype="S")

    try:
        stacked = np.stack(sanitized)
    except Exception:
        return json.dumps(sanitized, default=str)

    if stacked.dtype != object:
        return stacked

    return json.dumps(sanitized, default=str)


def sanitize_settings_value(value: Any) -> Any:
    """Convert a settings value into a tabular-friendly scalar or array value."""
    if value is None:
        return "None"
    if isinstance(value, Enum):
        return sanitize_settings_value(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value if value.dtype != object else _sanitize_sequence_value(value.tolist())
    if isinstance(value, (bool, int, float, str, bytes)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return json.dumps({str(k): sanitize_settings_value(v) for k, v in value.items()}, default=str)
    if isinstance(value, (set, frozenset)):
        return _sanitize_sequence_value(sorted(value, key=repr))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return _sanitize_sequence_value(value)
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)


def flatten_component_settings(component_address: str, value: Any) -> dict[str, Any]:
    """Flatten and sanitize a component settings payload for tabular storage."""
    if isinstance(value, ez.Settings):
        flat = flatten_ez_settings(value)
    elif hasattr(value, "structured_value") and getattr(value, "structured_value") is not None:
        flat = _flatten_value(getattr(value, "structured_value"), "")
    elif hasattr(value, "repr_value") and isinstance(getattr(value, "repr_value"), dict):
        flat = _flatten_value(getattr(value, "repr_value"), "")
    else:
        flat = _flatten_value(value, "")

    return {
        sanitize_settings_column_name(f"{component_address}.{field_name}"): sanitize_settings_value(field_value)
        for field_name, field_value in flat.items()
    }


# ---------------------------------------------------------------------------
# Event message
# ---------------------------------------------------------------------------


class PipelineSettingsEventType(str, Enum):
    """Distinguishes startup snapshot rows from in-flight settings changes."""

    INITIAL = "INITIAL"
    UPDATED = "UPDATED"


_DEFAULT_TABLE_NAME = "settings_annotations"


@dataclass
class PipelineSettingsEvent:
    """One settings-change observation, ready to ship across the graph.

    Mirrors the JSON-friendly portion of ``ezmsg.core.SettingsChangedEvent``
    so the message survives any ezmsg transport without custom encoders.
    Sinks that want a tabular row call :meth:`flatten_for_table`; sinks that
    need native column types can read ``structured_value`` directly and
    project with the helpers in this module.
    """

    seq: int
    """Monotonic sequence number from the graph server."""

    timestamp: float
    """Wall-clock seconds when the change was observed."""

    component_address: str
    """Address of the component whose settings changed."""

    event_type: PipelineSettingsEventType
    """``INITIAL`` for startup-snapshot rows, ``UPDATED`` for live changes."""

    repr_value: typing.Union[dict[str, Any], str]
    """Human-readable settings dump (``dict`` for dataclass-shaped settings,
    ``repr()`` string fallback otherwise)."""

    structured_value: Optional[dict[str, Any]] = None
    """Parsed structured settings dump (``None`` if structure couldn't be
    derived; falls back to ``repr_value``)."""

    source_session_id: Optional[str] = None

    table_name: str = _DEFAULT_TABLE_NAME
    """Name of the target table/series the consuming sink should write into."""

    def flatten_for_table(self) -> dict[str, Any]:
        """Project this event into ``NWBPointRow``-compatible columns.

        Returns ``{"data": json_string}`` so the event can be appended to a
        ``pynwb.misc.AnnotationSeries`` (single-string-per-timestamp). The
        JSON payload self-describes the change with ``component``,
        ``event_type``, ``seq``, and the structured ``settings`` snapshot.
        """
        if self.structured_value is not None:
            payload: typing.Union[dict[str, Any], str] = self.structured_value
        elif isinstance(self.repr_value, dict):
            payload = self.repr_value
        else:
            payload = self.repr_value

        return {
            "data": json.dumps(
                {
                    "component": self.component_address,
                    "event_type": self.event_type.value,
                    "seq": self.seq,
                    "settings": payload,
                },
                default=str,
            ),
        }


# ---------------------------------------------------------------------------
# Producer
# ---------------------------------------------------------------------------


class PipelineSettingsProducerSettings(ez.Settings):
    """Settings for :class:`PipelineSettingsProducer`."""

    target_table: str = _DEFAULT_TABLE_NAME
    """Value to stamp onto each emitted event's ``table_name`` field."""

    scope_components: Optional[tuple[str, ...]] = None
    """If set, only emit events for these component addresses. ``None`` means
    watch every component the graph server reports. ``PipelineSettingsUnit``
    fills this in automatically by discovering its own session's components
    on initialize."""


@processor_state
class PipelineSettingsProducerState:
    """State for :class:`PipelineSettingsProducer`."""

    ctx: Optional[ez.GraphContext] = None
    queue: Optional[asyncio.Queue] = None
    watch_task: Optional[asyncio.Task] = None
    last_seq: int = 0
    scope: Optional[frozenset[str]] = None
    initialized: bool = False


def _map_event_type(upstream: typing.Any) -> PipelineSettingsEventType:
    """Translate an upstream ``SettingsEventType`` into our public enum."""
    name = getattr(upstream, "name", str(upstream))
    if name == "INITIAL_SETTINGS":
        return PipelineSettingsEventType.INITIAL
    return PipelineSettingsEventType.UPDATED


class PipelineSettingsProducer(
    BaseStatefulProducer[
        PipelineSettingsProducerSettings,
        PipelineSettingsEvent,
        PipelineSettingsProducerState,
    ]
):
    """Stream the graph server's settings events out as :class:`PipelineSettingsEvent`.

    On reset (which happens on first ``__acall__``), the producer:

    1. Opens a ``GraphContext`` against the running graph server.
    2. Pulls the current settings snapshot and queues one ``INITIAL`` event
       per in-scope component.
    3. Subscribes to subsequent settings events; each one becomes an
       ``UPDATED`` event in the queue.

    ``_produce`` returns one event per call, awaiting the queue when it's
    empty. If the GraphContext can't be opened (no server running, network
    issue), the producer logs a warning and the queue stays empty — the unit
    will simply never publish anything, matching the "best-effort" semantics
    of the previous in-sink approach.
    """

    def _reset_state(self) -> None:
        # Producer needs an event loop to open the GraphContext; the real
        # work lives in _areset_state, which the framework calls from the
        # async __acall__ path.
        raise NotImplementedError("PipelineSettingsProducer requires async setup; call via __acall__.")

    async def _areset_state(self) -> None:
        # Tear down anything left over from a prior incarnation (settings-
        # driven reset).
        await self._teardown()

        self._state.queue = asyncio.Queue()
        self._state.last_seq = 0
        self._state.scope = (
            frozenset(self.settings.scope_components) if self.settings.scope_components is not None else None
        )
        self._state.initialized = False

        try:
            ctx = ez.GraphContext(auto_start=False)
            await ctx.__aenter__()
        except Exception as exc:
            ez.logger.warning(f"PipelineSettingsProducer could not open GraphContext: {exc}")
            return
        self._state.ctx = ctx

        try:
            await self._seed_initial_events()
            self._state.watch_task = asyncio.create_task(
                self._watch(after_seq=self._state.last_seq),
                name="pipeline-settings-watch",
            )
            self._state.initialized = True
        except Exception as exc:
            ez.logger.warning(f"PipelineSettingsProducer initial snapshot failed: {exc}")
            await self._teardown()

    async def _seed_initial_events(self) -> None:
        """Fill the queue with one ``INITIAL`` event per in-scope component."""
        ctx = self._state.ctx
        if ctx is None or self._state.queue is None:
            return

        settings_snapshot = await ctx.settings_snapshot()
        seed_events = await ctx.settings_events(after_seq=0)

        # For each component in the snapshot, find the most recent seed event
        # so the INITIAL row carries an accurate seq + timestamp.
        latest_per_addr: dict[str, typing.Any] = {}
        for ev in seed_events:
            cur = latest_per_addr.get(ev.component_address)
            if cur is None or ev.seq > cur.seq:
                latest_per_addr[ev.component_address] = ev

        scope = self._state.scope
        for addr in sorted(settings_snapshot.keys()):
            if scope is not None and addr not in scope:
                continue
            value = settings_snapshot[addr]
            latest = latest_per_addr.get(addr)
            seq = latest.seq if latest is not None else 0
            ts = latest.timestamp if latest is not None else time.time()
            session_id = latest.source_session_id if latest is not None else None
            self._state.queue.put_nowait(
                PipelineSettingsEvent(
                    seq=seq,
                    timestamp=ts,
                    component_address=addr,
                    event_type=PipelineSettingsEventType.INITIAL,
                    repr_value=value.repr_value,
                    structured_value=value.structured_value,
                    source_session_id=session_id,
                    table_name=self.settings.target_table,
                )
            )
            self._state.last_seq = max(self._state.last_seq, seq)

        self._state.last_seq = max(
            self._state.last_seq,
            max((ev.seq for ev in seed_events), default=0),
        )

    async def _watch(self, after_seq: int) -> None:
        """Forward live settings events into the queue until cancelled."""
        ctx = self._state.ctx
        queue = self._state.queue
        if ctx is None or queue is None:
            return
        scope = self._state.scope
        try:
            async for event in ctx.subscribe_settings_events(after_seq=after_seq):
                if scope is not None and event.component_address not in scope:
                    continue
                await queue.put(
                    PipelineSettingsEvent(
                        seq=event.seq,
                        timestamp=event.timestamp,
                        component_address=event.component_address,
                        event_type=_map_event_type(event.event_type),
                        repr_value=event.value.repr_value,
                        structured_value=event.value.structured_value,
                        source_session_id=event.source_session_id,
                        table_name=self.settings.target_table,
                    )
                )
                self._state.last_seq = event.seq
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            ez.logger.warning(f"PipelineSettingsProducer watch terminated: {exc}")

    async def _produce(self) -> PipelineSettingsEvent:
        # If reset failed to bring up a queue, block indefinitely so the unit
        # doesn't spin emitting Nones. The framework will cancel us on
        # shutdown.
        if self._state.queue is None:
            await asyncio.Event().wait()
            raise RuntimeError("unreachable")
        return await self._state.queue.get()

    async def _teardown(self) -> None:
        """Cancel the watch task and close the GraphContext."""
        task = self._state.watch_task
        self._state.watch_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        ctx = self._state.ctx
        self._state.ctx = None
        if ctx is not None:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass

        self._state.queue = None
        self._state.initialized = False


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


class PipelineSettingsUnit(
    BaseProducerUnit[
        PipelineSettingsProducerSettings,
        PipelineSettingsEvent,
        PipelineSettingsProducer,
    ]
):
    """Producer unit that emits one :class:`PipelineSettingsEvent` per change.

    Wire its ``OUTPUT_SIGNAL`` to any sink that consumes
    :class:`PipelineSettingsEvent` (or, more loosely, any sink that accepts
    objects implementing the ``NWBPointRow`` protocol — settings events
    satisfy it via :meth:`PipelineSettingsEvent.flatten_for_table`).

    On ``initialize``, if ``scope_components`` is unset, the unit looks up
    its own session in the graph snapshot and scopes the producer to that
    session's components. This matches the behaviour of the original
    in-sink watcher: only events from components running alongside the
    sink are emitted, so multiple parallel pipelines don't cross-pollute.
    """

    SETTINGS = PipelineSettingsProducerSettings

    async def initialize(self) -> None:
        # If the user didn't pre-supply a scope, discover this unit's session
        # and apply it to the producer's settings before reset runs.
        if self.SETTINGS.scope_components is None:
            try:
                scope = await self._discover_session_scope()
            except Exception as exc:
                ez.logger.warning(f"{self.address} could not discover session scope; will watch all components: {exc}")
                scope = None
            if scope:
                self.apply_settings(
                    PipelineSettingsProducerSettings(
                        target_table=self.SETTINGS.target_table,
                        scope_components=tuple(sorted(scope)),
                    )
                )
        await super().initialize()

    async def shutdown(self) -> None:
        prod = getattr(self, "producer", None)
        if prod is not None:
            try:
                await prod._teardown()
            except Exception:
                pass
        await super().shutdown()

    async def _discover_session_scope(self) -> Optional[set[str]]:
        ctx = ez.GraphContext(auto_start=False)
        await ctx.__aenter__()
        try:
            snapshot = await ctx.snapshot()
            for session in snapshot.sessions.values():
                metadata = session.metadata
                if metadata is None:
                    continue
                if self.address in metadata.components:
                    return set(metadata.components.keys())
        finally:
            await ctx.__aexit__(None, None, None)
        return None
