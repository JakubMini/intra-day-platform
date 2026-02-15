from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from intraday_platform.infrastructure.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    try:
        import streamlit  # noqa: F401
    except ImportError as exc:  # pragma: no cover - runtime check
        logger.error("Streamlit is not installed. Install the 'prod' deps and retry.", extra={"error": str(exc)})
        raise SystemExit(1) from exc

    app_path = Path(__file__).resolve().parent / "presentation" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    logger.info("Launching Streamlit app", extra={"command": " ".join(cmd)})
    try:
        raise SystemExit(subprocess.call(cmd))
    except KeyboardInterrupt:
        logger.info("Streamlit app interrupted by user")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
