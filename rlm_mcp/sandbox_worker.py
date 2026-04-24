from __future__ import annotations

import json
import os
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, unquote


@dataclass
class Chunk:
    chunk_id: str
    start_line: int
    end_line: int
    text: str
    preview: str


@dataclass
class ContextRecord:
    context_id: str
    uri: str
    path: str
    line_count: int
    char_count: int
    chunk_lines: int
    overlap_lines: int
    chunks: list[Chunk]
    lines: list[str]


STATE: dict[str, Any] = {
    "contexts": {},
    "active_context_id": None,
}


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _file_path_from_uri(uri: str) -> Path:
    if uri.startswith("file://"):
        parsed = urlparse(uri)
        return Path(unquote(parsed.path)).expanduser().resolve()
    return Path(uri).expanduser().resolve()


def _chunk_lines(lines: list[str], chunk_lines: int, overlap_lines: int) -> list[Chunk]:
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be positive")
    if overlap_lines < 0:
        raise ValueError("overlap_lines must be non-negative")
    step = max(1, chunk_lines - overlap_lines)
    chunks: list[Chunk] = []
    chunk_index = 0
    for start in range(0, len(lines), step):
        stop = min(len(lines), start + chunk_lines)
        chunk_text = "".join(lines[start:stop])
        preview = chunk_text.strip().splitlines()
        preview_text = preview[0][:180] if preview else ""
        chunks.append(
            Chunk(
                chunk_id=f"chunk-{chunk_index}",
                start_line=start + 1,
                end_line=stop,
                text=chunk_text,
                preview=preview_text,
            )
        )
        chunk_index += 1
        if stop >= len(lines):
            break
    return chunks


def _load_context(params: dict[str, Any]) -> dict[str, Any]:
    uri = params["uri"]
    chunk_lines = int(params.get("chunk_lines", 200))
    overlap_lines = int(params.get("overlap_lines", 20))

    path = _file_path_from_uri(uri)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    chunks = _chunk_lines(lines, chunk_lines, overlap_lines)
    context_id = f"{path.name}:{int(path.stat().st_mtime_ns)}"

    record = ContextRecord(
        context_id=context_id,
        uri=uri,
        path=str(path),
        line_count=len(lines),
        char_count=len(text),
        chunk_lines=chunk_lines,
        overlap_lines=overlap_lines,
        chunks=chunks,
        lines=lines,
    )
    STATE["contexts"][context_id] = record
    STATE["active_context_id"] = context_id

    return {
        "context_id": context_id,
        "path": str(path),
        "line_count": record.line_count,
        "char_count": record.char_count,
        "chunk_lines": chunk_lines,
        "overlap_lines": overlap_lines,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "preview": chunk.preview,
            }
            for chunk in chunks
        ],
    }


def _get_context(context_id: str | None) -> ContextRecord:
    resolved_id = context_id or STATE["active_context_id"]
    if not resolved_id:
        raise ValueError("No context loaded. Call rlm_load_context first.")
    record = STATE["contexts"].get(resolved_id)
    if not record:
        raise KeyError(f"Unknown context_id: {resolved_id}")
    return record


def _find_chunk(record: ContextRecord, chunk_id: str) -> Chunk:
    for chunk in record.chunks:
        if chunk.chunk_id == chunk_id:
            return chunk
    raise KeyError(f"Unknown chunk_id: {chunk_id}")


def _search_context(params: dict[str, Any]) -> dict[str, Any]:
    record = _get_context(params.get("context_id"))
    query = params["query"]
    regex = bool(params.get("regex", False))
    case_sensitive = bool(params.get("case_sensitive", False))
    max_results = int(params.get("max_results", 20))
    context_lines = max(0, int(params.get("context_lines", 1)))
    chunk_id = params.get("chunk_id")

    line_offset = 0
    candidate_lines = record.lines
    active_chunk = None
    if chunk_id:
        active_chunk = _find_chunk(record, chunk_id)
        line_offset = active_chunk.start_line - 1
        candidate_lines = record.lines[active_chunk.start_line - 1 : active_chunk.end_line]

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query, flags) if regex else None

    matches: list[dict[str, Any]] = []
    for index, line in enumerate(candidate_lines, start=1):
        haystack = line if case_sensitive else line.lower()
        needle = query if case_sensitive else query.lower()
        found = bool(pattern.search(line)) if pattern else needle in haystack
        if not found:
            continue
        absolute_line = line_offset + index
        start = max(1, absolute_line - context_lines)
        end = min(record.line_count, absolute_line + context_lines)
        snippet = "".join(record.lines[start - 1 : end])
        owning_chunk = next(
            (
                chunk.chunk_id
                for chunk in record.chunks
                if chunk.start_line <= absolute_line <= chunk.end_line
            ),
            None,
        )
        matches.append(
            {
                "line_number": absolute_line,
                "line": line.rstrip("\n"),
                "snippet_start_line": start,
                "snippet_end_line": end,
                "snippet": snippet,
                "chunk_id": owning_chunk,
            }
        )
        if len(matches) >= max_results:
            break

    return {
        "context_id": record.context_id,
        "query": query,
        "regex": regex,
        "case_sensitive": case_sensitive,
        "searched_chunk_id": active_chunk.chunk_id if active_chunk else None,
        "matches": matches,
        "match_count": len(matches),
    }


def _get_chunk_payload(params: dict[str, Any]) -> dict[str, Any]:
    record = _get_context(params.get("context_id"))
    chunk = _find_chunk(record, params["chunk_id"])
    include_neighbors = max(0, int(params.get("include_neighbor_chunks", 0)))

    selected: list[dict[str, Any]] = []
    chunk_ids = [c.chunk_id for c in record.chunks]
    center_index = chunk_ids.index(chunk.chunk_id)
    start_index = max(0, center_index - include_neighbors)
    end_index = min(len(record.chunks), center_index + include_neighbors + 1)

    for item in record.chunks[start_index:end_index]:
        selected.append(
            {
                "chunk_id": item.chunk_id,
                "start_line": item.start_line,
                "end_line": item.end_line,
                "preview": item.preview,
                "text": item.text,
            }
        )

    return {
        "context_id": record.context_id,
        "path": record.path,
        "target_chunk_id": chunk.chunk_id,
        "target_chunk": asdict(chunk),
        "selected_chunks": selected,
    }


COMMANDS = {
    "load_context": _load_context,
    "search_context": _search_context,
    "get_chunk_payload": _get_chunk_payload,
}


def main() -> int:
    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            message = json.loads(raw_line)
            command = message["command"]
            params = message.get("params", {})
            if command == "ping":
                _emit({"ok": True, "result": {"status": "ok", "pid": os.getpid()}})
                continue
            handler = COMMANDS.get(command)
            if not handler:
                raise ValueError(f"Unknown command: {command}")
            _emit({"ok": True, "result": handler(params)})
        except Exception as exc:  # noqa: BLE001
            _emit(
                {
                    "ok": False,
                    "error": {
                        "message": str(exc),
                        "type": exc.__class__.__name__,
                        "traceback": traceback.format_exc(),
                    },
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
