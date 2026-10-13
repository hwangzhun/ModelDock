"""Generate a headless screenshot with accurate CJK widths in Rich's SVG export."""
import asyncio
import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Use terminal colors for documentation even when invoked from a NO_COLOR shell.
os.environ.pop("NO_COLOR", None)
os.environ["TERM"] = "xterm-256color"
os.environ["COLORTERM"] = "truecolor"

from rich.cells import cell_len  # noqa: E402

from autodl_tui.app import AutoDLApp  # noqa: E402
from autodl_tui.config import Config  # noqa: E402


async def main():
    destination = Path(__file__).resolve().parents[1] / "docs"
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="autodl-preview-") as directory:
        app = AutoDLApp(Config(root=directory), Path(directory) / "config.json")
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause(1)
            app.save_screenshot("preview.svg", path=str(destination))
    path = destination / "preview.svg"
    # Rich 15.0 SVG uses len(text) for textLength, even though x offsets use cell_len.
    # Correct only the exported SVG, preserving actual UI layout and screenshot content.
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    root = ET.fromstring(path.read_text())
    for text in root.iter("{http://www.w3.org/2000/svg}text"):
        if "textLength" in text.attrib:
            text.set("textLength", str(12.2 * cell_len(text.text or "")))
    style = root.find("{http://www.w3.org/2000/svg}style")
    style.text = re.sub(r"@font-face\s*\{.*?\}", "", style.text, flags=re.S)
    style.text = style.text.replace("Fira Code, monospace", '"DejaVu Sans Mono", "Noto Sans Mono CJK SC", monospace')
    path.write_text(ET.tostring(root, encoding="unicode"))
    print(path)


if __name__ == "__main__":
    asyncio.run(main())
