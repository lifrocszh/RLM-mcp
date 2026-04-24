from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class MCPClient:
    def __init__(self, server_cmd: list[str]) -> None:
        self.proc = subprocess.Popen(
            server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
        )
        self.request_id = 0

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def _send(self, message: dict[str, Any]) -> dict[str, Any]:
        if not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("MCP server stdio is unavailable")
        body = json.dumps(message).encode("utf-8")
        self.proc.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
        self.proc.stdin.write(body)
        self.proc.stdin.flush()

        header = b""
        while True:
            line = self.proc.stdout.readline()
            if not line:
                stderr = b""
                if self.proc.stderr:
                    stderr = self.proc.stderr.read()
                raise RuntimeError(stderr.decode("utf-8", errors="replace") or "MCP server exited")
            if line in (b"\r\n", b"\n"):
                break
            header += line

        headers: dict[str, str] = {}
        for raw_line in header.decode("utf-8").splitlines():
            key, _, value = raw_line.partition(":")
            headers[key.strip().lower()] = value.strip()

        length = int(headers["content-length"])
        payload = self.proc.stdout.read(length)
        response = json.loads(payload.decode("utf-8"))
        if "error" in response:
            raise RuntimeError(json.dumps(response["error"], indent=2))
        return response["result"]

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.request_id += 1
        return self._send(
            {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params,
            }
        )

    def initialize(self) -> dict[str, Any]:
        return self.request("initialize", {})

    def list_tools(self) -> dict[str, Any]:
        return self.request("tools/list", {})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local smoke-test client for the RLM MCP server.")
    parser.add_argument("file", nargs="?", help="Optional file to load with rlm_load_context.")
    parser.add_argument("--query", help="Optional search query to run with rlm_search after loading a file.")
    parser.add_argument("--chunk-lines", type=int, default=200, help="Chunk size for rlm_load_context.")
    parser.add_argument("--overlap-lines", type=int, default=20, help="Overlap size for rlm_load_context.")
    parser.add_argument("--list-tools", action="store_true", help="List the server tools after initialize.")
    parser.add_argument(
        "--server-cmd",
        nargs="+",
        default=[sys.executable, "-m", "rlm_mcp.server"],
        help="Command used to start the MCP server process.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    client = MCPClient(args.server_cmd)
    try:
        init_result = client.initialize()
        print("initialize:")
        print(json.dumps(init_result, indent=2))

        if args.list_tools:
            tools_result = client.list_tools()
            print("\ntools/list:")
            print(json.dumps(tools_result, indent=2))

        if args.file:
            file_path = str(Path(args.file).expanduser().resolve())
            load_result = client.call_tool(
                "rlm_load_context",
                {
                    "uri": file_path,
                    "chunk_lines": args.chunk_lines,
                    "overlap_lines": args.overlap_lines,
                },
            )
            print("\nrlm_load_context:")
            print(json.dumps(load_result, indent=2))

            if args.query:
                search_result = client.call_tool("rlm_search", {"query": args.query})
                print("\nrlm_search:")
                print(json.dumps(search_result, indent=2))

        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
