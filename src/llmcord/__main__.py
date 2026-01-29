"""Package entry point for llmcord."""
from __future__ import annotations

import asyncio
import contextlib

from llmcord.runtime import main

with contextlib.suppress(KeyboardInterrupt):
    asyncio.run(main())
