"""Example usage of ezmsg-example package."""

import asyncio

import ezmsg.core as ez

# Import your units from the package
# from ezmsg.example import MyUnit


class ExampleSettings(ez.Settings):
    """Settings for ExampleUnit."""

    message: str = "Hello from ezmsg-example!"


class ExampleUnit(ez.Unit):
    """Example ezmsg Unit demonstrating basic patterns."""

    SETTINGS = ExampleSettings

    INPUT = ez.InputStream(str)
    OUTPUT = ez.OutputStream(str)

    @ez.subscriber(INPUT)
    @ez.publisher(OUTPUT)
    async def on_message(self, message: str) -> ez.AsyncGenerator:
        """Process incoming messages."""
        result = f"{self.SETTINGS.message} Received: {message}"
        yield self.OUTPUT, result


async def main():
    """Run the example."""
    print("ezmsg-example loaded successfully!")
    print(f"Version: {__import__('ezmsg.example').__version__}")

    # Example: Create and run a simple system
    # system = ExampleSystem()
    # await ez.run(SYSTEM=system)


if __name__ == "__main__":
    asyncio.run(main())
