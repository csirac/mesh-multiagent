## Worker Tool Instructions

Use the `mesh-tool` CLI to access mesh services. It is on your PATH.

```bash
mesh-tool                              # list all tools
mesh-tool <name>                       # show usage for one tool
mesh-tool <name> --arg1 val1 --arg2 val2   # call a tool (returns JSON)
```

Exit 0 + JSON on success, exit 1 + error on failure. Use absolute paths
(e.g. `/home/youruser/…`); tilde expansion is unreliable. Do NOT replicate mesh
services with Bash/curl/Python scripts.

### Priority tools — web research

| Tool | When to use |
|------|-------------|
| `exa_search` | General web search; returns snippets and URLs |
| `exa_fetch_full` | Fetch full page content from a URL |
| `literature_search` | Search academic literature (arXiv, PubMed, Semantic Scholar) |
| `literature_fulltext` | Get full text of a paper by ID |
| `arxiv_search` | Search arXiv specifically |
| `arxiv_get` | Fetch arXiv paper metadata by ID |

### Memory tools

| Tool | When to use |
|------|-------------|
| `memory_get` | Get full details of a memory entry by ID |
| `memory_search` | Search the agent's memory pool by keyword |
| `history_search` | Keyword search over raw conversation logs (lossless recall) |

### Essay tools

| Tool | When to use |
|------|-------------|
| `essay_get` | Read an interpretive essay by entity key |
| `essay_list` | List all interpretive essays |

### Standard harness file tools

| Tool | When to use |
|------|-------------|
| `file_read` | Read a file with line numbers |
| `file_edit` | Exact string replacement in a file |
| `apply_patch` | Multi-hunk or multi-file edits with context anchoring |
| `shell` | Execute a shell command |
| `list_dir` | List directory contents as a tree |
| `find_files` | Find files matching a glob pattern |
| `grep` | Search file contents for a regex pattern |
