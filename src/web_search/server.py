from __future__ import annotations

import os
import signal
import sys
import threading
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP

from .guide import build_doctor_report, build_usage_guide, recommend_command as build_recommendation

mcp = FastMCP("web-search")


@mcp.tool(
    name="get_usage_guide",
    output_schema=None,
    description=(
        "Describe how this repository should be used from bash. This MCP does not run web search itself; "
        "it teaches the model to execute `uv run web-search-cli ...` commands from the repository root, "
        "including the summary-bearing `searchwithsummary` command whose summary should be verified."
    ),
    meta={"version": "3.0.0", "author": "guda.studio"},
)
def get_usage_guide() -> dict:
    return build_usage_guide()


@mcp.tool(
    name="recommend_command",
    output_schema=None,
    description=(
        "Given a natural-language task, recommend the exact `uv run web-search-cli ...` command the model "
        "should execute in bash. Returns a fallback command list when the first command is not enough, and "
        "marks `searchwithsummary` as draft-only when that variant is recommended."
    ),
    meta={"version": "3.0.0", "author": "guda.studio"},
)
def recommend_command(
    task: Annotated[str, "Natural-language task or research request."],
) -> dict:
    return build_recommendation(task)


@mcp.tool(
    name="doctor",
    output_schema=None,
    description=(
        "Check whether the repository is ready for bash-driven search. Returns config status, connection test, "
        "and the next command the model should run."
    ),
    meta={"version": "3.0.0", "author": "guda.studio"},
)
async def doctor() -> dict:
    return await build_doctor_report()


def _install_signal_handlers() -> None:
    if threading.current_thread() is not threading.main_thread():
        return

    def handle_shutdown(signum, frame):
        os._exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, handle_shutdown)


def _start_windows_parent_monitor() -> None:
    if sys.platform != "win32":
        return
    import ctypes
    import time

    parent_pid = os.getppid()

    def is_parent_alive(pid: int) -> bool:
        process_query = 0x1000
        still_active = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(process_query, False, pid)
        if not handle:
            return True
        exit_code = ctypes.c_ulong()
        result = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return result and exit_code.value == still_active

    def monitor_parent() -> None:
        while True:
            if not is_parent_alive(parent_pid):
                os._exit(0)
            time.sleep(2)

    threading.Thread(target=monitor_parent, daemon=True).start()


def main() -> None:
    src_dir = Path(__file__).parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    _install_signal_handlers()
    _start_windows_parent_monitor()
    try:
        mcp.run(transport="stdio", show_banner=False)
    except KeyboardInterrupt:
        pass
    finally:
        os._exit(0)


if __name__ == "__main__":
    main()
