"""The mechanism packages use to deprecate a per-processor ``axis`` setting.

The policy -- which settings go away, and in which release -- belongs to each
package. What lives here is the machinery: a warning that fires once per call
site, names the caller rather than the library, and can be suppressed while a
stage forwards a setting to a child it built itself.

The settings class below stands in for a real one. ``ez.Settings`` are frozen
dataclasses, so a ``__post_init__`` is reached by every construction path.
"""

import inspect
import warnings

import ezmsg.core as ez

from ezmsg.baseproc import suppress_axis_deprecation, warn_axis_deprecated


class ExampleSettings(ez.Settings):
    axis: str | None = None
    other: int = 0

    def __post_init__(self) -> None:
        warn_axis_deprecated(self, package="ezmsg-example", removal="9.9")


class RenamedFieldSettings(ez.Settings):
    preserve_axis: str | None = None

    def __post_init__(self) -> None:
        warn_axis_deprecated(self, "preserve_axis")


def axis_warnings(records):
    return [r for r in records if issubclass(r.category, FutureWarning) and "deprecated" in str(r.message)]


class TestItWarnsOnlyWhenSet:
    def test_setting_it_warns(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ExampleSettings(axis="time")
        assert len(axis_warnings(rec)) == 1

    def test_leaving_it_unset_is_silent(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ExampleSettings(other=3)
        assert not axis_warnings(rec)

    def test_passing_none_explicitly_is_also_silent(self):
        """``None`` is the value that means "follow the stream", so asking for it
        is not a use of the deprecated behaviour."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ExampleSettings(axis=None)
        assert not axis_warnings(rec)

    def test_it_is_a_futurewarning(self):
        """DeprecationWarning is suppressed by default outside __main__, so a
        pipeline -- library code -- would never see it."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ExampleSettings(axis="time")
        assert rec[0].category is FutureWarning

    def test_a_differently_named_field_works(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            RenamedFieldSettings(preserve_axis="time")
        (record,) = axis_warnings(rec)
        assert "RenamedFieldSettings.preserve_axis" in str(record.message)


class TestTheMessageIsActionable:
    def test_it_names_the_package_and_release(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ExampleSettings(axis="time")
        message = str(axis_warnings(rec)[0].message)
        assert "ezmsg-example 9.9" in message

    def test_an_unnamed_release_falls_back_rather_than_lying(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            RenamedFieldSettings(preserve_axis="time")
        assert "a future release" in str(axis_warnings(rec)[0].message)


# A settings class that genuinely lives in an `ezmsg.*` module, which the class
# above cannot: defined in the test file, its `__post_init__` is indistinguishable
# from user code and the walk correctly stops there. Built through `exec` with
# planted globals because the frame test keys on `f_globals["__name__"]` -- the
# dataclass `__init__` is itself exec-generated, so `co_filename` is "<string>"
# for library and `python -c` alike and cannot be the discriminator.
_LIBRARY_GLOBALS = {
    "__name__": "ezmsg.example.stage",
    "ez": ez,
    "warn_axis_deprecated": warn_axis_deprecated,
}
exec(
    "class LibrarySettings(ez.Settings):\n"
    "    axis: str | None = None\n"
    "\n"
    "    def __post_init__(self) -> None:\n"
    "        warn_axis_deprecated(self, package='ezmsg-example', removal='9.9')\n"
    "\n"
    "\n"
    "def library_factory(axis):\n"
    "    return LibrarySettings(axis=axis)\n",
    _LIBRARY_GLOBALS,
)
LibrarySettings = _LIBRARY_GLOBALS["LibrarySettings"]
library_factory = _LIBRARY_GLOBALS["library_factory"]


class TestItBlamesTheCaller:
    def test_the_stacklevel_points_at_the_construction_site(self):
        """The whole point: the user sees their own line, not ours."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            LibrarySettings(axis="time")
            expected_line = inspect.currentframe().f_lineno - 1
        (record,) = axis_warnings(rec)
        assert (record.filename, record.lineno) == (__file__, expected_line)

    def test_intervening_library_frames_are_skipped(self):
        """A factory that builds the settings object inside the library must not
        be blamed for a call the user made -- the case a fixed stacklevel gets
        wrong even when it is right for direct construction."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            library_factory("time")
            expected_line = inspect.currentframe().f_lineno - 1
        (record,) = axis_warnings(rec)
        assert (record.filename, record.lineno) == (__file__, expected_line)

    def test_a_class_defined_outside_ezmsg_blames_its_own_post_init(self):
        """The flip side, recorded so the behaviour is not mistaken for a bug:
        the walk cannot skip a settings class that does not live under
        ``ezmsg.``, because nothing distinguishes it from the caller."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            ExampleSettings(axis="time")
        (record,) = axis_warnings(rec)
        assert record.filename == __file__


class TestSuppression:
    def test_it_silences_and_restores(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with suppress_axis_deprecation():
                ExampleSettings(axis="time")
            assert not axis_warnings(rec)
            ExampleSettings(axis="time")
        assert len(axis_warnings(rec)) == 1

    def test_it_restores_even_when_the_body_raises(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                with suppress_axis_deprecation():
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            ExampleSettings(axis="time")
        assert len(axis_warnings(rec)) == 1

    def test_it_nests(self):
        """A collection forwarding into a child that forwards again."""
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            with suppress_axis_deprecation():
                with suppress_axis_deprecation():
                    ExampleSettings(axis="time")
                ExampleSettings(axis="time")
            assert not axis_warnings(rec)
            ExampleSettings(axis="time")
        assert len(axis_warnings(rec)) == 1
