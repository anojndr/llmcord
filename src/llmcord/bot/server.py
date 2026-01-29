"""Health check web server for the bot."""
import os

from aiohttp import web

from llmcord.config import get_config, get_health_check_port


async def health_check(_request: web.Request) -> web.Response:
    """Return a basic liveness response."""
    return web.Response(text="I'm alive")


async def start_server() -> None:
    """Start a small HTTP health-check server."""
    app = web.Application()
    app.add_routes([web.get("/", health_check)])
    runner = web.AppRunner(app)
    await runner.setup()
    config = get_config()
    port = get_health_check_port(config)
    if port is None:
        port = int(os.environ.get("PORT", "8000"))
    host = config.get("health_check_host") or os.environ.get("HOST", "127.0.0.1")
    site = web.TCPSite(runner, host, port)
    await site.start()
