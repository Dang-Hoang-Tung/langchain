from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Optional


def open_png_bytes(
    png_bytes: bytes,
    *,
    filename: Optional[str] = None,
    keep: bool = False,
) -> Path:
    """
    Save PNG bytes to a file and open it in the system image viewer.

    Args:
        png_bytes: Raw PNG bytes.
        filename: Optional filename (defaults to a temp file).
        keep: If True, do not delete the file on exit.

    Returns:
        Path to the written PNG file.
    """
    if filename:
        path = Path(filename).expanduser().resolve()
        path.write_bytes(png_bytes)
    else:
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        path = Path(tmp)
        path.write_bytes(png_bytes)

    if sys.platform.startswith("darwin"):       # macOS
        os.system(f"open {path}")
    elif sys.platform.startswith("linux"):      # Linux
        os.system(f"xdg-open {path}")
    elif sys.platform.startswith("win"):        # Windows
        os.system(f"start {path}")
    else:
        print(f"Image saved to: {path}")

    return path
