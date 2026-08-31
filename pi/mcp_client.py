"""Minimal MCP client (stdio transport, JSON-RPC 2.0).

The pipeline uses this to pull real case context from connected clinical systems (EHR,
OR board, formulary) exposed as MCP servers, instead of inferring everything from the
transcript. Servers are declared in a config file:

    { "mcpServers": { "<name>": { "command": "...", "args": [...], "env": {...} } } }

Config path: PI_MCP_CONFIG, else ./.mcp.json. If neither exists, the pool is empty and
the pipeline falls back to transcript-only context.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional

_PROTOCOL = "2025-06-18"


class MCPServer:
    """One stdio MCP server subprocess."""

    def __init__(self, name: str, command: str, args: list[str], env: Optional[dict] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = {**os.environ, **(env or {})}
        self.proc: Optional[asyncio.subprocess.Process] = None
        self.tools: list[dict] = []
        self._id = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=self.env,
        )
        await self._request("initialize", {
            "protocolVersion": _PROTOCOL,
            "capabilities": {},
            "clientInfo": {"name": "procedural-intelligence", "version": "0.1.0"},
        })
        await self._notify("notifications/initialized", {})
        res = await self._request("tools/list", {})
        self.tools = (res or {}).get("tools", [])

    async def call(self, tool: str, arguments: dict) -> Any:
        res = await self._request("tools/call", {"name": tool, "arguments": arguments})
        for block in (res or {}).get("content", []):
            if block.get("type") == "text":
                try:
                    return json.loads(block["text"])
                except json.JSONDecodeError:
                    return block["text"]
        return res

    async def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            try:
                self.proc.stdin.close()
                await asyncio.wait_for(self.proc.wait(), timeout=3)
            except Exception:  # noqa: BLE001
                self.proc.kill()

    # ---- transport ----
    async def _send(self, obj: dict) -> None:
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        await self.proc.stdin.drain()

    async def _notify(self, method: str, params: dict) -> None:
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _request(self, method: str, params: dict) -> Optional[dict]:
        async with self._lock:
            self._id += 1
            rid = self._id
            await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            while True:
                raw = await asyncio.wait_for(self.proc.stdout.readline(), timeout=20)
                if not raw:
                    raise RuntimeError(f"MCP server {self.name!r} closed the stream")
                try:
                    msg = json.loads(raw.decode())
                except json.JSONDecodeError:
                    continue
                if msg.get("id") == rid:
                    if "error" in msg:
                        raise RuntimeError(f"MCP {self.name}.{method}: {msg['error']}")
                    return msg.get("result")


class MCPPool:
    def __init__(self, servers: list[MCPServer]):
        self.servers = servers

    @property
    def enabled(self) -> bool:
        return bool(self.servers)

    async def __aenter__(self) -> "MCPPool":
        for s in list(self.servers):
            try:
                await s.start()
            except Exception as exc:  # noqa: BLE001
                print(f"  [mcp] {s.name} unavailable: {exc}")
                self.servers.remove(s)
        return self

    async def __aexit__(self, *_exc) -> None:
        await asyncio.gather(*(s.stop() for s in self.servers), return_exceptions=True)

    def find_tool(self, name: str) -> Optional[MCPServer]:
        return next((s for s in self.servers if any(t["name"] == name for t in s.tools)), None)

    async def call(self, tool: str, arguments: dict) -> Any:
        s = self.find_tool(tool)
        if s is None:
            raise KeyError(f"no connected MCP server exposes {tool!r}")
        return await s.call(tool, arguments)

    def has(self, tool: str) -> bool:
        return self.find_tool(tool) is not None


def _config_path() -> Optional[Path]:
    p = os.environ.get("PI_MCP_CONFIG")
    if p:
        return Path(p)
    default = Path.cwd() / ".mcp.json"
    return default if default.exists() else None


def load_pool() -> MCPPool:
    if os.environ.get("PI_MCP", "").lower() in {"0", "off", "false", "no"}:
        return MCPPool([])
    cfg_path = _config_path()
    if not cfg_path or not cfg_path.exists():
        return MCPPool([])
    cfg = json.loads(cfg_path.read_text())
    servers = []
    for name, spec in (cfg.get("mcpServers") or {}).items():
        servers.append(MCPServer(name, spec["command"], spec.get("args", []), spec.get("env")))
    return MCPPool(servers)
