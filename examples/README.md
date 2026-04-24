# Examples

This directory contains reproducible examples for the RLM MCP server.

## Benchmark

`benchmark_mcp_vs_flat.py` compares:

- flat file reading, where the full file is assumed to be sent into model context for every query
- the MCP workflow, where the file is loaded once into the sandbox and subsequent queries use `rlm_search`

Run it with:

```bash
cd RLM-mcp
uv run python examples/benchmark_mcp_vs_flat.py
```

If you are in an offline or restricted environment and already have the project available locally:

```bash
cd RLM-mcp
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python examples/benchmark_mcp_vs_flat.py
```

Outputs:

- `benchmark_corpus.txt`: the generated large-file fixture
- `benchmark_results.json`: machine-readable metrics
- `benchmark_report.md`: a GitHub-friendly summary
