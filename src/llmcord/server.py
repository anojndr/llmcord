"""Simple health-check server for llmcord."""

import os

from aiohttp import web

from llmcord.core.config import get_config


async def health_check(_request: web.Request) -> web.Response:
    """Return a basic liveness response."""
    return web.Response(text="I'm alive")


async def start_server() -> None:
    """Start a small HTTP health-check server."""
    config = get_config()
    app = web.Application()
    app.add_routes([web.get("/", health_check)])
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", str(config.get("port", 8001))))
    host = os.environ.get("HOST", "0.0.0.0")  # noqa: S104
    site = web.TCPSite(runner, host, port)
    await site.start()
