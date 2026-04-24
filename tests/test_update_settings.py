"""Tests for BaseProcessor / BaseProducer live settings updates."""

import dataclasses

import pytest

from ezmsg.baseproc import (
    BaseProcessor,
    BaseProducer,
    BaseStatefulProcessor,
    BaseStatefulProducer,
    CompositeProcessor,
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


@dataclasses.dataclass
class CompSettings:
    stage1_window: int = 10
    stage2_threshold: float = 0.5
    include_extra: bool = False


@processor_state
class CountingState:
    resets: int = 0
    closes: int = 0
    hash: int = -1


class CountingChild(BaseStatefulProcessor[UpdSettings, MsgA, MsgA, CountingState]):
    """Identity-passing stateful child; `threshold` is NONRESET-safe."""

    NONRESET_SETTINGS_FIELDS = frozenset({"threshold"})

    def _reset_state(self, message: MsgA) -> None:
        self._state.resets += 1

    def _process(self, message: MsgA) -> MsgA:
        return message

    def close(self) -> None:
        self._state.closes += 1


class _PipelineComposite(CompositeProcessor[CompSettings, MsgA, MsgA]):
    """Two- or three-stage composite driven by CompSettings."""

    @staticmethod
    def _initialize_processors(settings: CompSettings) -> dict[str, object]:
        procs: dict[str, object] = {
            "stage1": CountingChild(settings=UpdSettings(window_size=settings.stage1_window)),
            "stage2": CountingChild(settings=UpdSettings(threshold=settings.stage2_threshold)),
        }
        if settings.include_extra:
            procs["extra"] = CountingChild(settings=UpdSettings())
        return procs


class TestCompositeUpdateSettings:
    """Default CompositeProcessor.update_settings reconciliation rules."""

    def test_matching_children_preserve_state_via_delegated_update(self):
        comp = _PipelineComposite(settings=CompSettings())
        comp(MsgA())
        stage1_before = comp._procs["stage1"]
        stage2_before = comp._procs["stage2"]
        assert stage1_before.state.resets == 1
        assert stage2_before.state.resets == 1

        # stage2_threshold is NONRESET-safe on the child; stage1_window is not.
        comp.update_settings(CompSettings(stage1_window=20, stage2_threshold=0.9))

        # Same instances survived (type-match + update_settings path).
        assert comp._procs["stage1"] is stage1_before
        assert comp._procs["stage2"] is stage2_before

        # stage1 sees a non-safe field change → reset queued on child.
        assert stage1_before._hash == -1
        # stage2 only saw a safe field change → no reset queued.
        assert stage2_before._hash != -1

        comp(MsgA())
        assert stage1_before.state.resets == 2
        assert stage2_before.state.resets == 1

    def test_new_key_added_when_flag_flips_on(self):
        comp = _PipelineComposite(settings=CompSettings(include_extra=False))
        comp(MsgA())
        assert "extra" not in comp._procs

        comp.update_settings(CompSettings(include_extra=True))
        assert "extra" in comp._procs
        # Pipeline still processes; extra runs its first reset.
        comp(MsgA())
        assert comp._procs["extra"].state.resets == 1

    def test_disappeared_key_is_closed(self):
        comp = _PipelineComposite(settings=CompSettings(include_extra=True))
        comp(MsgA())
        extra = comp._procs["extra"]

        comp.update_settings(CompSettings(include_extra=False))
        assert "extra" not in comp._procs
        assert extra.state.closes == 1

    def test_no_op_update_skips_rebuild(self):
        comp = _PipelineComposite(settings=CompSettings())
        comp(MsgA())
        stage1_before = comp._procs["stage1"]

        # Identical settings → no children touched.
        comp.update_settings(CompSettings())
        assert comp._procs["stage1"] is stage1_before
        # Confirm no spurious reset was requested on the child.
        assert stage1_before._hash != -1

    def test_composite_nonreset_fields_skip_rebuild(self):
        class AlmostNoopComposite(_PipelineComposite):
            NONRESET_SETTINGS_FIELDS = frozenset({"stage1_window"})

        comp = AlmostNoopComposite(settings=CompSettings())
        comp(MsgA())
        stage1_before = comp._procs["stage1"]

        # Change a NONRESET-safe field on the composite: no rebuild, no child touch.
        comp.update_settings(CompSettings(stage1_window=999))
        assert comp._procs["stage1"] is stage1_before
        assert stage1_before.settings.window_size == 10  # child untouched
        assert comp.settings.stage1_window == 999  # composite did rebind
