from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JSONRPC_VERSION = "2.0"


class MCPError(RuntimeError):
    """Protocol-layer error."""


@dataclass
class ToolResult:
    text: str
    structured: dict[str, Any] | None = None
    is_error: bool = False


class SandboxClient:
    def __init__(self) -> None:
        backend = os.getenv("RLM_SANDBOX_BACKEND", "local").strip().lower()
        self.backend = backend
        self.proc = self._spawn(backend)
        self._rpc("ping", {})

    def _spawn(self, backend: str) -> subprocess.Popen[str]:
        cwd = Path(__file__).resolve().parents[1]
        if backend == "local":
            cmd = [sys.executable, "-m", "rlm_mcp.sandbox_worker"]
        elif backend == "docker":
            image = os.getenv("RLM_DOCKER_IMAGE", "python:3.12-slim")
            cmd = [
                "docker",
                "run",
                "--rm",
                "-i",
                "-v",
                f"{cwd}:/workspace",
                "-w",
                "/workspace",
                image,
                "python",
                "-m",
                "rlm_mcp.sandbox_worker",
            ]
        elif backend == "modal":
            raise NotImplementedError(
                "Modal sandbox backend is not implemented in this version."
            )
        else:
            raise ValueError(f"Unsupported sandbox backend: {backend}")

        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(cwd),
        )

    def _rpc(self, command: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("Sandbox process is unavailable.")
        self.proc.stdin.write(json.dumps({"command": command, "params": params}) + "\n")
        self.proc.stdin.flush()
        raw = self.proc.stdout.readline()
        if not raw:
            stderr = ""
            if self.proc.stderr:
                stderr = self.proc.stderr.read().strip()
            raise RuntimeError(f"Sandbox exited unexpectedly. stderr={stderr}")
        payload = json.loads(raw)
        if not payload.get("ok"):
            error = payload.get("error", {})
            raise RuntimeError(error.get("message", "Sandbox call failed"))
        return payload["result"]

    def load_context(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("load_context", params)

    def search(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("search_context", params)

    def chunk_payload(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("get_chunk_payload", params)


class WorkerModelClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.model = os.getenv("RLM_MODEL", "gpt-4o-mini")
        self.system_prompt = os.getenv(
            "RLM_WORKER_SYSTEM_PROMPT",
            (
                "You are a recursive worker model operating on a focused chunk from a much larger file. "
                "Answer only from the provided chunk payload. If the evidence is insufficient, say so clearly "
                "and explain what additional chunk or search would help."
            ),
        )

    def completion(self, sub_query: str, chunk_payload: dict[str, Any], temperature: float) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for rlm_recursive_call")

        body = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "sub_query": sub_query,
                            "chunk_payload": chunk_payload,
                        },
                        ensure_ascii=True,
                    ),
                },
            ],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Worker model HTTP error: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Worker model network error: {exc}") from exc

        message = payload["choices"][0]["message"]["content"]
        if isinstance(message, list):
            text = "".join(part.get("text", "") for part in message if isinstance(part, dict))
        else:
            text = str(message)

        return {
            "model": self.model,
            "answer": text.strip(),
            "raw_response_id": payload.get("id"),
        }


class MCPServer:
    def __init__(self) -> None:
        self.sandbox = SandboxClient()
        self.worker = WorkerModelClient()
        self.server_info = {
            "name": "rlm-mcp",
            "version": "0.1.0",
        }

    def run(self) -> int:
        while True:
            message = self._read_message()
            if message is None:
                return 0
            if "id" not in message:
                self._handle_notification(message)
                continue
            response = self._handle_request(message)
            self._write_message(response)

    def _read_message(self) -> dict[str, Any] | None:
        header_bytes = b""
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            header_bytes += line

        headers = {}
        for raw_line in header_bytes.decode("utf-8").splitlines():
            key, _, value = raw_line.partition(":")
            headers[key.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", "0"))
        if content_length <= 0:
            raise MCPError("Missing or invalid Content-Length header")

        body = sys.stdin.buffer.read(content_length)
        if not body:
            return None
        return json.loads(body.decode("utf-8"))

    def _write_message(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(body)
        sys.stdout.buffer.flush()

    def _handle_notification(self, message: dict[str, Any]) -> None:
        if message.get("method") == "notifications/initialized":
            return

    def _handle_request(self, message: dict[str, Any]) -> dict[str, Any]:
        request_id = message["id"]
        method = message.get("method")
        params = message.get("params", {})
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": self.server_info,
                }
            elif method == "tools/list":
                result = {"tools": self._tool_definitions()}
            elif method == "tools/call":
                result = self._call_tool(params)
            else:
                raise MCPError(f"Unsupported method: {method}")
            return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}
        except Exception as exc:  # noqa: BLE001
            return {
                "jsonrpc": JSONRPC_VERSION,
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": str(exc),
                    "data": {"type": exc.__class__.__name__},
                },
            }

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "rlm_load_context",
                "description": (
                    "Load a large file into a persistent sandbox, chunk it, and return metadata "
                    "without sending the file contents through the model context."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "uri": {"type": "string"},
                        "chunk_lines": {"type": "integer", "default": 200},
                        "overlap_lines": {"type": "integer", "default": 20},
                    },
                    "required": ["uri"],
                },
            },
            {
                "name": "rlm_search",
                "description": (
                    "Run substring or regex search against sandbox-resident context and return "
                    "line-level hits with chunk references."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "regex": {"type": "boolean", "default": False},
                        "case_sensitive": {"type": "boolean", "default": False},
                        "context_id": {"type": "string"},
                        "chunk_id": {"type": "string"},
                        "max_results": {"type": "integer", "default": 20},
                        "context_lines": {"type": "integer", "default": 1},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "rlm_recursive_call",
                "description": (
                    "Spawn a focused worker model call over a chunk stored in the sandbox. "
                    "This is the MCP-side adaptation of an RLM Sub_RLM recursive call."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "sub_query": {"type": "string"},
                        "chunk_id": {"type": "string"},
                        "context_id": {"type": "string"},
                        "include_neighbor_chunks": {"type": "integer", "default": 0},
                        "temperature": {"type": "number", "default": 0.0},
                    },
                    "required": ["sub_query", "chunk_id"],
                },
            },
        ]

    def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name == "rlm_load_context":
            outcome = self._tool_load_context(arguments)
        elif name == "rlm_search":
            outcome = self._tool_search(arguments)
        elif name == "rlm_recursive_call":
            outcome = self._tool_recursive_call(arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

        result: dict[str, Any] = {
            "content": [{"type": "text", "text": outcome.text}],
            "isError": outcome.is_error,
        }
        if outcome.structured is not None:
            result["structuredContent"] = outcome.structured
        return result

    def _tool_load_context(self, arguments: dict[str, Any]) -> ToolResult:
        data = self.sandbox.load_context(arguments)
        summary = (
            f"Loaded {data['path']} into sandbox as {data['context_id']} with "
            f"{data['line_count']} lines split into {data['chunk_count']} chunks."
        )
        return ToolResult(text=summary, structured=data)

    def _tool_search(self, arguments: dict[str, Any]) -> ToolResult:
        data = self.sandbox.search(arguments)
        if not data["matches"]:
            summary = f"No matches found for {data['query']!r} in context {data['context_id']}."
        else:
            lines = [
                f"{item['chunk_id']}:{item['line_number']}: {item['line']}"
                for item in data["matches"][:10]
            ]
            summary = "\n".join(lines)
        return ToolResult(text=summary, structured=data)

    def _tool_recursive_call(self, arguments: dict[str, Any]) -> ToolResult:
        chunk_payload = self.sandbox.chunk_payload(arguments)
        temperature = float(arguments.get("temperature", 0.0))
        worker_result = self.worker.completion(
            sub_query=arguments["sub_query"],
            chunk_payload=chunk_payload,
            temperature=temperature,
        )
        data = {
            "backend": self.sandbox.backend,
            "worker_model": worker_result["model"],
            "context_id": chunk_payload["context_id"],
            "target_chunk_id": chunk_payload["target_chunk_id"],
            "selected_chunks": [
                {
                    "chunk_id": item["chunk_id"],
                    "start_line": item["start_line"],
                    "end_line": item["end_line"],
                    "preview": item["preview"],
                }
                for item in chunk_payload["selected_chunks"]
            ],
            "answer": worker_result["answer"],
            "raw_response_id": worker_result["raw_response_id"],
        }
        return ToolResult(text=worker_result["answer"], structured=data)


def main() -> int:
    server = MCPServer()
    return server.run()


if __name__ == "__main__":
    raise SystemExit(main())
