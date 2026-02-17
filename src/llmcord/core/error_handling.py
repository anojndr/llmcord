"""Centralized exception logging and process-level hooks."""

from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Mapping
    from types import TracebackType

LOGGER = logging.getLogger(__name__)
COMMON_HANDLER_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _format_context(context: Mapping[str, object]) -> str:
    """Render structured context as a stable key-value string."""
    return ", ".join(f"{key}={context[key]!r}" for key in sorted(context))


def log_exception(
    *,
    logger: logging.Logger,
    message: str,
    error: BaseException,
    context: Mapping[str, object] | None = None,
) -> None:
    """Write a structured exception log entry."""
    if context:
        logger.error(
            "%s | %s",
            message,
            _format_context(context),
            exc_info=error,
        )
        return
    logger.error(message, exc_info=error)


def register_asyncio_exception_handler(
    loop: asyncio.AbstractEventLoop,
    *,
    logger: logging.Logger | None = None,
) -> None:
    """Install a loop-level asyncio exception handler."""
    target_logger = logger or LOGGER

    def _handle_exception(
        _loop: asyncio.AbstractEventLoop,
        context: dict[str, Any],
    ) -> None:
        error = context.get("exception")
        message = str(context.get("message") or "Unhandled asyncio exception")
        if isinstance(error, BaseException):
            log_exception(
                logger=target_logger,
                message=message,
                error=error,
                context=context,
            )
            return
        target_logger.error("%s | %s", message, _format_context(context))

    loop.set_exception_handler(_handle_exception)


def install_global_exception_hooks(*, logger: logging.Logger | None = None) -> None:
    """Install process-level exception hooks for uncaught exceptions."""
    target_logger = logger or LOGGER
    previous_excepthook = sys.excepthook
    previous_thread_hook = threading.excepthook

    def _sys_excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous_excepthook(exc_type, exc_value, exc_traceback)
            return
        target_logger.error(
            "Unhandled exception at process boundary",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        if issubclass(args.exc_type, KeyboardInterrupt):
            previous_thread_hook(args)
            return
        thread_name = args.thread.name if args.thread else "unknown"
        exc_value = args.exc_value
        if not isinstance(exc_value, BaseException):
            target_logger.error("Unhandled thread exception | thread=%s", thread_name)
            return
        target_logger.error(
            "Unhandled thread exception | thread=%s",
            thread_name,
            exc_info=(args.exc_type, exc_value, args.exc_traceback),
        )

    sys.excepthook = _sys_excepthook
    threading.excepthook = _thread_excepthook


def log_discord_event_error(
    *,
    logger: logging.Logger,
    event_name: str,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> None:
    """Log uncaught Discord event exceptions with event metadata."""
    exc_value = sys.exc_info()[1]
    context = {
        "event": event_name,
        "args_count": len(args),
        "kwargs_keys": tuple(sorted(kwargs)),
    }
    if isinstance(exc_value, BaseException):
        log_exception(
            logger=logger,
            message="Unhandled Discord event error",
            error=exc_value,
            context=context,
        )
        return
    logger.error("Unhandled Discord event error | %s", _format_context(context))
