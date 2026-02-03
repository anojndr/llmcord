"""Main entry point for running the llmcord application."""

import asyncio

import llmcord.entrypoint


def main() -> None:
    """Run the application entry point."""
    try:
        asyncio.run(llmcord.entrypoint.main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
