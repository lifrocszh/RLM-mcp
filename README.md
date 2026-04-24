# RLM MCP Server

This project implements a Model Context Protocol (MCP) server that adapts the main ideas from the MIT paper *Recursive Language Models* by Alex L. Zhang, Tim Kraska, and Omar Khattab (`arXiv:2512.24601`).

Instead of pushing an entire file into the model context, this server keeps large files inside a persistent Python sandbox and exposes focused tools:

- `rlm_load_context(uri)` loads a file into a persistent sandbox process and builds chunk metadata without sending file contents to the model.
- `rlm_search(query, ...)` performs regex or substring search inside the sandbox over the loaded context.
- `rlm_recursive_call(sub_query, chunk_id, ...)` materializes a small sub-problem from sandbox state and sends only that focused slice to a configured worker model.

The design follows the paper's core pattern: treat long context as an external environment, let the model inspect it programmatically, and support recursive sub-calls over smaller views of the original input.

Reference material:

- Paper: <https://arxiv.org/abs/2512.24601>
- MIT write-up: <https://alexzhang13.github.io/blog/2025/rlm/>
- Official codebase: <https://github.com/alexzhang13/rlm>

## Architecture

The server is split into two processes:

1. `rlm_mcp.server` speaks MCP over stdio using JSON-RPC with `Content-Length` framing.
2. `rlm_mcp.sandbox_worker` is a persistent Python sandbox process that stores loaded context and responds to structured commands.

That separation keeps the large file in an external environment and prevents accidental re-serialization into the host model prompt.

## Sandbox Backends

`RLM_SANDBOX_BACKEND` controls how the persistent sandbox runs:

- `local` (default): launches the sandbox worker as a local child process.
- `docker`: launches the sandbox worker inside Docker for stronger isolation.
- `modal`: reserved for future work; the code raises a clear "not implemented" error today.

Docker mode expects a working Docker CLI and mounts this repository into the container.

## Worker Model Configuration

`rlm_recursive_call` uses an OpenAI-compatible chat endpoint and standard library HTTP calls. Configure it with:

- `OPENAI_API_KEY`: required for worker calls
- `RLM_MODEL`: model name for recursive worker calls, default `gpt-4o-mini`
- `OPENAI_BASE_URL`: optional, default `https://api.openai.com/v1`
- `RLM_WORKER_SYSTEM_PROMPT`: optional custom system prompt

The recursive worker only receives:

- the sub-query
- chunk metadata
- the selected chunk text
- optional nearby chunks when requested

It does not receive the entire original file.

## Running

Use `uv` as the default workflow:

```bash
cd RLM-mcp
uv sync
uv run python -m rlm_mcp.server
```

Or use the script entrypoint:

```bash
cd RLM-mcp
uv sync
uv run rlm-mcp
```

If you need a shell inside the project environment:

```bash
cd RLM-mcp
uv sync
uv shell
```

## Quick Smoke Test

Start the server in one terminal:

```bash
cd RLM-mcp
uv run python -m rlm_mcp.server
```

Then run the local client script from another terminal:

```bash
cd RLM-mcp
uv run python scripts/mcp_client.py --list-tools
```

Or load a file and search it in a single command:

```bash
cd RLM-mcp
uv run python scripts/mcp_client.py README.md --query "rlm_search"
```

## Evaluation

The repository includes a reproducible benchmark in [examples/README.md](examples/README.md) that compares:

- flat full-file prompting
- the RLM MCP workflow using `rlm_load_context` and `rlm_search`

Current benchmark findings from `examples/benchmark_report.md`:

- Dataset size: `124,607` characters, `2,160` lines, about `31,151` estimated full-file tokens
- Flat baseline across 3 benchmark queries: `93,525` estimated tokens
- MCP workflow including one-time load: `3,606` estimated tokens
- Estimated token reduction: `96.14%`
- Average `rlm_search` latency: `0.68 ms`

Per-query reduction from the current run:

- `RLM_NOTE_173`: `99.39%`
- `SEARCH_KEY_219`: `99.49%`
- `recursive summarization anchor`: `91.34%`

## Running The Examples

Run the benchmark:

```bash
cd RLM-mcp
uv run python examples/benchmark_mcp_vs_flat.py
```

If you are in an offline or restricted environment and already have the repo set up locally:

```bash
cd RLM-mcp
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python examples/benchmark_mcp_vs_flat.py
```

Generated outputs:

- `examples/benchmark_corpus.txt`
- `examples/benchmark_results.json`
- `examples/benchmark_report.md`

You can also inspect the example-specific documentation:

```bash
cd RLM-mcp
cat examples/README.md
cat examples/benchmark_report.md
```

## Using With Zed

Zed supports MCP servers as custom context servers. Open the Agent Panel settings with `agent: open settings`, then add a server entry under `context_servers`.

Example configuration:

```json
{
  "context_servers": {
    "rlm-mcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/RLM-mcp",
        "python",
        "-m",
        "rlm_mcp.server"
      ],
      "env": {
        "RLM_SANDBOX_BACKEND": "local",
        "OPENAI_API_KEY": "your-api-key-if-you-want-rlm_recursive_call"
      }
    }
  }
}
```

Notes:

- Replace `"/absolute/path/to/RLM-mcp"` with the actual cloned repo path on your machine.
- Run `uv sync` in the repo first so the environment is ready.
- If you only want `rlm_load_context` and `rlm_search`, you can omit `OPENAI_API_KEY`.
- `OPENAI_API_KEY` is only needed for `rlm_recursive_call`, because that tool makes its own outbound model request from inside the MCP server process.
- Zed's built-in AI provider configuration is separate from custom MCP server environment variables. Adding an OpenAI key to Zed lets Zed call models, but it does not automatically inject that key into arbitrary MCP subprocesses.
- If you launch Zed from a shell where `OPENAI_API_KEY` is already exported, the MCP server may inherit it through the process environment and you can omit it from the JSON.
- If Zed was launched from the Dock, launcher, or another environment that does not already contain `OPENAI_API_KEY`, set it explicitly in `context_servers.rlm-mcp.env` or avoid `rlm_recursive_call`.
- In Zed's Agent settings, a green status dot next to the server means the MCP process started successfully.
- If you want Zed to stop prompting for every MCP tool call, set `agent.tool_permissions.default` to `"allow"` or add per-tool permissions for `mcp:rlm-mcp:...`.

Once the server is active, mention `rlm-mcp` in the prompt and ask the agent to use `rlm_load_context`, `rlm_search`, or `rlm_recursive_call`.

## MCP Tools

### `rlm_load_context`

Arguments:

- `uri`: file path or `file://` URI
- `chunk_lines`: optional, default `200`
- `overlap_lines`: optional, default `20`

Returns:

- context id
- file metadata
- chunk metadata and previews

### `rlm_search`

Arguments:

- `query`: search string or regex
- `regex`: optional, default `false`
- `case_sensitive`: optional, default `false`
- `context_id`: optional, uses most recently loaded context
- `chunk_id`: optional, search only one chunk
- `max_results`: optional, default `20`
- `context_lines`: optional, default `1`

Returns matching lines, line ranges, and chunk references.

### `rlm_recursive_call`

Arguments:

- `sub_query`: focused question for a worker model
- `chunk_id`: target chunk id
- `context_id`: optional, uses most recently loaded context
- `include_neighbor_chunks`: optional, default `0`
- `temperature`: optional, default `0.0`

Returns:

- worker answer
- chunk metadata
- backend/model metadata

## Notes

- This is a practical MCP-oriented adaptation of the RLM paper, not a full reimplementation of the paper's training or benchmark stack.
- The recursive behavior lives at inference time through tool-mediated decomposition and worker model calls over sandbox-resident context.
