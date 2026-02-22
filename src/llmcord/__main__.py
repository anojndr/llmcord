"""Main entry point for running the llmcord application."""

import asyncio

import llmcord.entrypoint
from llmcord.core.error_handling import (
    install_global_exception_hooks,
    register_asyncio_exception_handler,
)


def main() -> None:
    """Run the application entry point."""
    install_global_exception_hooks()
    with asyncio.Runner() as runner:
        register_asyncio_exception_handler(runner.get_loop())
        try:
            runner.run(llmcord.entrypoint.main())
        except KeyboardInterrupt:
            # Ensure the Discord client is closed before the event loop is torn down.
            runner.run(llmcord.entrypoint.shutdown())


if __name__ == "__main__":
    main()
