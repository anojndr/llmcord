"""Main entry point for running the llmcord application."""

import asyncio
import contextlib

import llmcord.entrypoint
from llmcord.core.error_handling import (
    install_global_exception_hooks,
    register_asyncio_exception_handler,
)


def main() -> None:
    """Run the application entry point."""
    install_global_exception_hooks()
    with contextlib.suppress(KeyboardInterrupt), asyncio.Runner() as runner:
        register_asyncio_exception_handler(runner.get_loop())
        runner.run(llmcord.entrypoint.main())


if __name__ == "__main__":
    main()
