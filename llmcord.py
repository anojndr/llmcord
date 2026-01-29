"""Root entry point for running llmcord without installing the package."""
from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "src"
if SRC_PATH.is_dir():
    sys.path.insert(0, str(SRC_PATH))

from llmcord.runtime import main  # noqa: E402

with contextlib.suppress(KeyboardInterrupt):
    asyncio.run(main())
