"""pytest configuration for ezmsg-baseproc tests."""

import os
import sys

import pytest

# Add tests directory to path so 'tests.helpers' can be imported
_tests_dir = os.path.dirname(__file__)
_parent_dir = os.path.dirname(_tests_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


@pytest.fixture
def test_name(request):
    """Provide the test name to test functions."""
    return request.node.name
