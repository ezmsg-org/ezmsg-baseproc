"""Example tests for ezmsg-example package."""


def test_import():
    """Test that the package can be imported."""
    import ezmsg.example

    assert hasattr(ezmsg.example, "__version__")


def test_version():
    """Test that version is a string."""
    from ezmsg.example import __version__

    assert isinstance(__version__, str)


# Add your tests below
# Example async test (requires pytest-asyncio):
#
# @pytest.mark.asyncio
# async def test_async_example():
#     """Example async test."""
#     assert True
