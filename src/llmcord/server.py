"""Simple health-check server for llmcord."""

import logging
import os
from collections.abc import Awaitable, Callable

from aiohttp import web

from llmcord.core.config import get_config
from llmcord.core.error_handling import COMMON_HANDLER_EXCEPTIONS, log_exception

logger = logging.getLogger(__name__)

RequestHandler = Callable[[web.Request], Awaitable[web.StreamResponse]]
SERVER_HANDLER_EXCEPTIONS = COMMON_HANDLER_EXCEPTIONS


@web.middleware
async def _error_middleware(
    request: web.Request,
    handler: RequestHandler,
) -> web.StreamResponse:
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except SERVER_HANDLER_EXCEPTIONS as exc:
        log_exception(
            logger=logger,
            message="Unhandled health-check server error",
            error=exc,
            context={
                "method": request.method,
                "path": request.path,
            },
        )
        return web.Response(status=500, text="Internal server error")


async def health_check(_request: web.Request) -> web.Response:
    """Return a basic liveness response."""
    return web.Response(text="I'm alive")


async def start_server() -> web.AppRunner:
    """Start a small HTTP health-check server.

    Returns the underlying aiohttp runner so callers can clean it up on shutdown.
    """
    config = get_config()
    app = web.Application(middlewares=[_error_middleware])
    app.add_routes([web.get("/", health_check)])
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", str(config.get("port", 8001))))
    host = os.environ.get("HOST", config.get("host"))
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner
