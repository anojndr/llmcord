"""Main entry point for running the llmcord application."""

import asyncio
import contextlib

import llmcord.entrypoint


def main() -> None:
    """Run the application entry point."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(llmcord.entrypoint.main())


if __name__ == "__main__":
    main()
