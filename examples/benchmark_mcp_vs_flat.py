from __future__ import annotations

import json
import statistics
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mcp_client import MCPClient

EXAMPLES_DIR = ROOT / "examples"
DATASET_PATH = EXAMPLES_DIR / "benchmark_corpus.txt"
REPORT_PATH = EXAMPLES_DIR / "benchmark_report.md"
RESULTS_PATH = EXAMPLES_DIR / "benchmark_results.json"

CHUNK_LINES = 120
OVERLAP_LINES = 12


@dataclass
class QueryCase:
    name: str
    query: str
    expected_hits: int


def estimate_tokens(text: str) -> int:
    # Cheap, dependency-free approximation suitable for relative comparisons.
    return max(1, len(text) // 4)


def build_corpus() -> str:
    sections: list[str] = []
    for i in range(1, 241):
        repeated = " ".join(
            [
                f"service_{i % 9} emits checkpoint_{i % 5}",
                "The pipeline normalizes records before indexing.",
                "Background workers flush metrics every interval.",
                "This paragraph exists to increase file size for benchmark realism.",
            ]
        )
        sections.append(
            textwrap.dedent(
                f"""\
                ## Section {i}
                module_name = "module_{i}"
                owner = "team_{i % 7}"
                {repeated}
                RLM_NOTE_{i:03d}: recursive summarization anchor for chunk {i}.
                SEARCH_KEY_{i:03d}: special token for evaluation query routing.
                Error handling path {i} retries on timeout and stores an audit trail.
                The design keeps file reading flat in baseline mode and indexed in MCP mode.
                """
            )
        )
    return "\n".join(sections)


def ensure_dataset() -> str:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    corpus = build_corpus()
    DATASET_PATH.write_text(corpus, encoding="utf-8")
    return corpus


def flatten_tool_text(result: dict[str, Any]) -> str:
    content = result.get("content", [])
    return "\n".join(item.get("text", "") for item in content if item.get("type") == "text")


def run_benchmark() -> dict[str, Any]:
    corpus = ensure_dataset()
    full_file_tokens = estimate_tokens(corpus)
    queries = [
        QueryCase(name="rare-anchor", query="RLM_NOTE_173", expected_hits=1),
        QueryCase(name="search-key", query="SEARCH_KEY_219", expected_hits=1),
        QueryCase(name="shared-phrase", query="recursive summarization anchor", expected_hits=20),
    ]

    server_cmd = ["python3", "-m", "rlm_mcp.server"]
    client = MCPClient(server_cmd)
    started_at = time.perf_counter()
    initialize_result = client.initialize()
    init_seconds = time.perf_counter() - started_at

    load_started = time.perf_counter()
    load_result = client.call_tool(
        "rlm_load_context",
        {
            "uri": str(DATASET_PATH.resolve()),
            "chunk_lines": CHUNK_LINES,
            "overlap_lines": OVERLAP_LINES,
        },
    )
    load_seconds = time.perf_counter() - load_started

    load_text = flatten_tool_text(load_result)
    load_structured = load_result["structuredContent"]
    load_tokens = estimate_tokens(load_text + json.dumps(load_structured, sort_keys=True))

    query_rows: list[dict[str, Any]] = []
    search_latencies: list[float] = []
    flat_tokens_total = 0
    mcp_tokens_total = load_tokens

    try:
        for case in queries:
            baseline_prompt = (
                "You are answering a question about a source file.\n"
                f"Question: {case.query}\n"
                "File contents:\n"
                f"{corpus}"
            )
            flat_tokens = estimate_tokens(baseline_prompt)
            flat_tokens_total += flat_tokens

            search_started = time.perf_counter()
            search_result = client.call_tool(
                "rlm_search",
                {
                    "query": case.query,
                    "max_results": case.expected_hits,
                    "context_lines": 1,
                },
            )
            search_seconds = time.perf_counter() - search_started
            search_latencies.append(search_seconds)

            search_text = flatten_tool_text(search_result)
            search_structured = search_result["structuredContent"]
            mcp_tokens = estimate_tokens(search_text + json.dumps(search_structured, sort_keys=True))
            mcp_tokens_total += mcp_tokens

            query_rows.append(
                {
                    "name": case.name,
                    "query": case.query,
                    "flat_tokens": flat_tokens,
                    "mcp_tokens": mcp_tokens,
                    "token_reduction_percent": round(100 * (1 - (mcp_tokens / flat_tokens)), 2),
                    "search_latency_ms": round(search_seconds * 1000, 2),
                    "match_count": search_structured["match_count"],
                }
            )
    finally:
        client.close()

    return {
        "dataset": {
            "path": str(DATASET_PATH.relative_to(ROOT)),
            "characters": len(corpus),
            "estimated_full_file_tokens": full_file_tokens,
            "line_count": corpus.count("\n") + 1,
        },
        "server": {
            "initialize_seconds": round(init_seconds, 6),
            "load_context_seconds": round(load_seconds, 6),
            "initialize_result": initialize_result,
            "context_id": load_structured["context_id"],
            "chunk_count": load_structured["chunk_count"],
            "chunk_lines": load_structured["chunk_lines"],
            "overlap_lines": load_structured["overlap_lines"],
        },
        "totals": {
            "flat_tokens_for_queries": flat_tokens_total,
            "mcp_tokens_for_queries_plus_load": mcp_tokens_total,
            "token_reduction_percent": round(100 * (1 - (mcp_tokens_total / flat_tokens_total)), 2),
            "avg_search_latency_ms": round(statistics.mean(search_latencies) * 1000, 2),
            "p95_search_latency_ms": round(max(search_latencies) * 1000, 2),
        },
        "queries": query_rows,
    }


def write_report(results: dict[str, Any]) -> None:
    dataset = results["dataset"]
    totals = results["totals"]
    server = results["server"]
    query_rows = results["queries"]

    lines = [
        "# MCP Evaluation: Flat File Reading vs RLM MCP",
        "",
        "This benchmark compares two strategies on the same large synthetic file:",
        "",
        "- Flat baseline: every query sends the entire file to the model context.",
        "- RLM MCP workflow: load the file once into the sandbox, then answer queries via `rlm_search` results.",
        "",
        "## Dataset",
        "",
        f"- File: `{Path(dataset['path']).name}`",
        f"- Characters: `{dataset['characters']}`",
        f"- Lines: `{dataset['line_count']}`",
        f"- Estimated full-file tokens: `{dataset['estimated_full_file_tokens']}`",
        "",
        "## Server Overhead",
        "",
        f"- `initialize`: `{server['initialize_seconds']}` seconds",
        f"- `rlm_load_context`: `{server['load_context_seconds']}` seconds",
        f"- Chunks created: `{server['chunk_count']}` with `chunk_lines={server['chunk_lines']}` and `overlap_lines={server['overlap_lines']}`",
        "",
        "## Token Comparison",
        "",
        f"- Flat total for benchmark queries: `{totals['flat_tokens_for_queries']}` estimated tokens",
        f"- MCP total including one-time load: `{totals['mcp_tokens_for_queries_plus_load']}` estimated tokens",
        f"- Estimated token reduction: `{totals['token_reduction_percent']}%`",
        "",
        "## Query Results",
        "",
        "| Query | Flat Tokens | MCP Tokens | Reduction | Search Latency | Matches |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in query_rows:
        lines.append(
            f"| `{row['query']}` | {row['flat_tokens']} | {row['mcp_tokens']} | "
            f"{row['token_reduction_percent']}% | {row['search_latency_ms']} ms | {row['match_count']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Token counts are estimated with a simple `len(text) / 4` heuristic, so treat them as directional rather than provider-exact.",
            "- This benchmark measures the current server's local tool overhead and context footprint, not remote LLM completion quality or network latency.",
            "- `rlm_recursive_call` is excluded here because it requires a live model API key and would mix model latency with MCP transport costs.",
        ]
    )

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    results = run_benchmark()
    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    write_report(results)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {RESULTS_PATH}")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
