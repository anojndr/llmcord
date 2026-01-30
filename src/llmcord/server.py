import os

from aiohttp import web


async def health_check(_request: web.Request) -> web.Response:
    """Return a basic liveness response."""
    return web.Response(text="I'm alive")


async def start_server() -> None:
    """Start a small HTTP health-check server."""
    app = web.Application()
    app.add_routes([web.get("/", health_check)])
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    site = web.TCPSite(runner, host, port)
    await site.start()
