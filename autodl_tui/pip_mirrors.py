"""Small, dependency-free mirror benchmark, also usable before the TUI is installed."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Callable
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

MIRRORS = (
    ("阿里云", "https://mirrors.aliyun.com/pypi/simple"),
    ("腾讯云", "https://mirrors.cloud.tencent.com/pypi/simple"),
    ("清华", "https://pypi.tuna.tsinghua.edu.cn/simple"),
)
SAMPLE_FILE = "setuptools-75.8.0-py3-none-any.whl"
SAMPLE_BYTES = 256 * 1024
TIMEOUT = 4


class WheelLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href", "")
            if unquote(urlsplit(href).path).rsplit("/", 1)[-1] == SAMPLE_FILE:
                self.links.append(href)


@dataclass(frozen=True)
class Result:
    name: str
    url: str
    seconds: float = 0
    error: str = ""


def read_sample(response, limit: int, deadline: float) -> bytes:
    chunks = []
    remaining = limit
    while remaining:
        if time.monotonic() >= deadline:
            raise TimeoutError("测速超时")
        block = response.read1(min(64 * 1024, remaining))
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def probe(name: str, index: str) -> Result:
    """Time index lookup plus the same 256 KiB wheel sample on each mirror."""
    started = time.monotonic()
    deadline = started + TIMEOUT * 2
    headers = {"User-Agent": "ModelDock/pip-benchmark", "Accept-Encoding": "identity"}
    try:
        listing_url = index.rstrip("/") + "/setuptools/"
        with urlopen(Request(listing_url, headers=headers), timeout=TIMEOUT) as response:
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}")
            listing = read_sample(response, 2 * 1024 * 1024 + 1, deadline)
            if len(listing) > 2 * 1024 * 1024:
                raise ValueError("索引页面过大")
            parser = WheelLinks()
            parser.feed(listing.decode("utf-8", errors="replace"))
            if not parser.links:
                raise ValueError("索引缺少测速样本")
            wheel_url = urljoin(response.url, parser.links[0])
        parsed = urlsplit(wheel_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("样本下载地址无效")
        headers["Range"] = f"bytes=0-{SAMPLE_BYTES - 1}"
        with urlopen(Request(wheel_url, headers=headers), timeout=TIMEOUT) as response:
            if response.status not in (200, 206):
                raise ValueError(f"HTTP {response.status}")
            sample = read_sample(response, SAMPLE_BYTES, deadline)
            if len(sample) != SAMPLE_BYTES or not sample.startswith(b"PK\x03\x04"):
                raise ValueError("返回内容不是有效的 wheel 样本")
        return Result(name, index, max(time.monotonic() - started, 0.001))
    except (OSError, ValueError) as error:
        return Result(name, index, error=str(error))


def select_mirror(log: Callable[[str], None] = lambda line: print(line, file=sys.stderr)) -> str:
    log("并行测速 pip 源：阿里云 / 腾讯云 / 清华（每个源读取 256 KiB 包文件）")
    results = []
    with ThreadPoolExecutor(max_workers=len(MIRRORS)) as pool:
        pending = [pool.submit(probe, name, url) for name, url in MIRRORS]
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            if result.error:
                log(f"{result.name}：不可用（{result.error}）")
            else:
                log(f"{result.name}：{result.seconds:.2f} 秒，约 {SAMPLE_BYTES / result.seconds / 1024:.0f} KiB/s")
    available = [result for result in results if not result.error]
    if not available:
        raise RuntimeError("三个 pip 源均不可用，请检查网络或手动指定 pip 源")
    chosen = min(available, key=lambda result: (result.seconds, result.url))
    log(f"本次使用 pip 源：{chosen.name} {chosen.url}")
    return chosen.url


if __name__ == "__main__":
    try:
        print(select_mirror())
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
