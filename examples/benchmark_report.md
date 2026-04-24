# MCP Evaluation: Flat File Reading vs RLM MCP

This benchmark compares two strategies on the same large synthetic file:

- Flat baseline: every query sends the entire file to the model context.
- RLM MCP workflow: load the file once into the sandbox, then answer queries via `rlm_search` results.

## Dataset

- File: `benchmark_corpus.txt`
- Characters: `124607`
- Lines: `2160`
- Estimated full-file tokens: `31151`

## Server Overhead

- `initialize`: `0.106605` seconds
- `rlm_load_context`: `0.001317` seconds
- Chunks created: `20` with `chunk_lines=120` and `overlap_lines=12`

## Token Comparison

- Flat total for benchmark queries: `93525` estimated tokens
- MCP total including one-time load: `3606` estimated tokens
- Estimated token reduction: `96.14%`

## Query Results

| Query | Flat Tokens | MCP Tokens | Reduction | Search Latency | Matches |
| --- | ---: | ---: | ---: | ---: | ---: |
| `RLM_NOTE_173` | 31173 | 190 | 99.39% | 0.53 ms | 1 |
| `SEARCH_KEY_219` | 31174 | 159 | 99.49% | 0.62 ms | 1 |
| `recursive summarization anchor` | 31178 | 2701 | 91.34% | 0.39 ms | 20 |

## Notes

- Token counts are estimated with a simple `len(text) / 4` heuristic, so treat them as directional rather than provider-exact.
- This benchmark measures the current server's local tool overhead and context footprint, not remote LLM completion quality or network latency.
- `rlm_recursive_call` is excluded here because it requires a live model API key and would mix model latency with MCP transport costs.
