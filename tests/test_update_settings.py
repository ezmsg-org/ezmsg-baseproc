"""Tests for BaseProcessor / BaseProducer live settings updates."""

import dataclasses

import pytest

from ezmsg.baseproc import (
    BaseProcessor,
    BaseProducer,
    BaseStatefulProcessor,
    BaseStatefulProducer,
    processor_state,
)


@dataclasses.dataclass
class UpdSettings:
    threshold: float = 0.5
    window_size: int = 100
    label: str = "a"


@processor_state
class UpdState:
    resets: int = 0
    processed: int = 0
    last_threshold: float = 0.0
    hash: int = -1


class MsgA:
    pass


class MsgB:
    pass


class NonStatefulProc(BaseProcessor[UpdSettings, MsgA, MsgB]):
    # "label" is a display-only field; changing it should not force a reset.
    NONRESET_SETTINGS_FIELDS = frozenset({"label"})

    def _process(self, message: MsgA) -> MsgB:
        return MsgB()


class StatefulProc(BaseStatefulProcessor[UpdSettings, MsgA, MsgB, UpdState]):
    NONRESET_SETTINGS_FIELDS = frozenset({"threshold"})

    def _reset_state(self, message: MsgA) -> None:
        self._state.resets += 1
        self._state.last_threshold = self.settings.threshold

    def _process(self, message: MsgA) -> MsgB:
        self._state.processed += 1
        return MsgB()


class NonStatefulProducer(BaseProducer[UpdSettings, MsgA]):
    NONRESET_SETTINGS_FIELDS = frozenset({"label"})

    async def _produce(self) -> MsgA:
        return MsgA()


class StatefulProducer(BaseStatefulProducer[UpdSettings, MsgA, UpdState]):
    NONRESET_SETTINGS_FIELDS = frozenset({"threshold"})

    def _reset_state(self) -> None:
        self._state.resets += 1
        self._state.last_threshold = self.settings.threshold

    async def _produce(self) -> MsgA:
        self._state.processed += 1
        return MsgA()


class TestNonStatefulUpdateSettings:
    def test_updates_settings_attribute(self):
        proc = NonStatefulProc(settings=UpdSettings())
        proc.update_settings(UpdSettings(threshold=0.9, window_size=50, label="b"))
        assert proc.settings.threshold == 0.9
        assert proc.settings.window_size == 50
        assert proc.settings.label == "b"

    def test_no_state_to_reset_is_fine(self):
        proc = NonStatefulProc(settings=UpdSettings())
        # Non-stateful processor just swaps settings; no reset path exists.
        proc.update_settings(UpdSettings(threshold=0.1))
        # Subsequent processing uses new settings via self.settings.
        assert isinstance(proc(MsgA()), MsgB)


class TestStatefulProcessorUpdateSettings:
    def test_reset_triggered_by_non_safe_field(self):
        proc = StatefulProc(settings=UpdSettings(threshold=0.5, window_size=100))
        # Warm up: first call performs the initial reset.
        proc(MsgA())
        assert proc._hash == 0
        assert proc.state.resets == 1

        # Change window_size (not in NONRESET_SETTINGS_FIELDS) → reset queued.
        proc.update_settings(UpdSettings(threshold=0.5, window_size=200))
        assert proc._hash == -1  # reset flag armed
        assert proc.settings.window_size == 200

        # Next message triggers _reset_state before _process.
        proc(MsgA())
        assert proc.state.resets == 2
        assert proc._hash == 0  # back to the hash _hash_message() returns

    def test_safe_field_change_preserves_hash(self):
        proc = StatefulProc(settings=UpdSettings(threshold=0.5, window_size=100))
        proc(MsgA())
        assert proc.state.resets == 1

        # Only a NONRESET_SETTINGS_FIELDS field changed → no reset queued.
        proc.update_settings(UpdSettings(threshold=0.75, window_size=100))
        assert proc._hash == 0
        assert proc.settings.threshold == 0.75

        proc(MsgA())
        # No additional reset happened.
        assert proc.state.resets == 1
        assert proc.state.processed == 2

    def test_reset_sees_new_settings(self):
        """_reset_state must observe the already-updated self.settings."""
        proc = StatefulProc(settings=UpdSettings(threshold=0.5, window_size=100))
        proc(MsgA())
        assert proc.state.last_threshold == 0.5

        # Change a reset-triggering field AND threshold together.
        proc.update_settings(UpdSettings(threshold=0.9, window_size=200))
        proc(MsgA())
        assert proc.state.last_threshold == 0.9

    def test_no_op_update_does_nothing(self):
        proc = StatefulProc(settings=UpdSettings())
        proc(MsgA())
        proc.update_settings(UpdSettings())
        assert proc._hash == 0
        proc(MsgA())
        assert proc.state.resets == 1  # still just the warm-up reset

    def test_default_empty_nonreset_resets_on_any_change(self):
        class StrictProc(BaseStatefulProcessor[UpdSettings, MsgA, MsgB, UpdState]):
            def _reset_state(self, message: MsgA) -> None:
                self._state.resets += 1

            def _process(self, message: MsgA) -> MsgB:
                return MsgB()

        proc = StrictProc(settings=UpdSettings())
        proc(MsgA())
        assert proc.state.resets == 1

        # Any field change → reset, because NONRESET_SETTINGS_FIELDS default is empty.
        proc.update_settings(UpdSettings(label="b"))
        proc(MsgA())
        assert proc.state.resets == 2


class TestStatefulProducerUpdateSettings:
    @pytest.mark.asyncio
    async def test_reset_triggered_by_non_safe_field(self):
        prod = StatefulProducer(settings=UpdSettings(threshold=0.5, window_size=100))
        await prod.__acall__()
        assert prod.state.resets == 1

        prod.update_settings(UpdSettings(threshold=0.5, window_size=200))
        assert prod._hash == -1
        assert prod.settings.window_size == 200

        await prod.__acall__()
        assert prod.state.resets == 2

    @pytest.mark.asyncio
    async def test_safe_field_change_preserves_hash(self):
        prod = StatefulProducer(settings=UpdSettings(threshold=0.5, window_size=100))
        await prod.__acall__()
        prod.update_settings(UpdSettings(threshold=0.75, window_size=100))
        assert prod._hash == 0
        await prod.__acall__()
        assert prod.state.resets == 1

    @pytest.mark.asyncio
    async def test_non_stateful_producer_just_updates_settings(self):
        prod = NonStatefulProducer(settings=UpdSettings())
        prod.update_settings(UpdSettings(threshold=0.9))
        assert prod.settings.threshold == 0.9
        assert isinstance(await prod.__acall__(), MsgA)


class TestOverrideUpdateSettings:
    """Fine-grained control via update_settings override (criterion 3)."""

    def test_subclass_override_takes_full_control(self):
        class CustomProc(BaseStatefulProcessor[UpdSettings, MsgA, MsgB, UpdState]):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.override_called = 0

            def _reset_state(self, message: MsgA) -> None:
                self._state.resets += 1

            def _process(self, message: MsgA) -> MsgB:
                return MsgB()

            def update_settings(self, new_settings: UpdSettings) -> None:
                # Bespoke rule: reset only when window_size shrinks.
                self.override_called += 1
                old_window = self.settings.window_size
                self.settings = new_settings
                if new_settings.window_size < old_window:
                    self._request_reset()

        proc = CustomProc(settings=UpdSettings(window_size=100))
        proc(MsgA())
        assert proc.state.resets == 1

        # Grow → no reset.
        proc.update_settings(UpdSettings(window_size=200))
        proc(MsgA())
        assert proc.state.resets == 1
        assert proc.override_called == 1

        # Shrink → reset.
        proc.update_settings(UpdSettings(window_size=50))
        proc(MsgA())
        assert proc.state.resets == 2
