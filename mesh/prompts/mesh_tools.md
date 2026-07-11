## Mesh Tools (shell access)

You have access to mesh tools via the `mesh-tool` command line utility.
These tools provide email, calendar, notes, web search, memory, literature search,
and inter-agent communication capabilities.

### Usage

```bash
# List all available tools
mesh-tool

# Show usage for a specific tool
mesh-tool <name>

# Call a tool (returns JSON to stdout)
mesh-tool <name> --arg1 value1 --arg2 value2
```

### Examples

```bash
mesh-tool gmail_list_recent --limit 10
mesh-tool gmail_list_unread --limit 5
mesh-tool gmail_search_emails --query "from:user subject:deploy" --limit 5
mesh-tool notes_search --query "mesh architecture" --db personal
mesh-tool exa_search --query "submodular optimization survey" --num_results 3
mesh-tool memory_search --query "router restart incident"
mesh-tool memory_get --id m_xxxx
mesh-tool current_time
```

### Gmail account switching: `--account`

All Gmail tools accept an `--account` flag to select which email account
to use (`work` or `personal`). If omitted, defaults to `work`.

```bash
# Read personal inbox
mesh-tool gmail_list_recent --account personal --limit 5

# Read work inbox (default)
mesh-tool gmail_list_recent --limit 5

# Search personal email
mesh-tool gmail_search_emails --account personal --query "from:someone" --limit 10

# Send from personal account
mesh-tool gmail_send_message --account personal --to "x@y.com" --subject "Hi" --body "Hello"
```

**Important:** Each `mesh-tool` call runs in a fresh subprocess — there is
no persistent account state between calls. Always pass `--account` on every
call that needs a non-default account. Do NOT use `account_set_current`
followed by separate Gmail calls (the state won't carry over).

### Shell-safety: dollar signs and special characters

When passing dollar amounts (`$550`), backticks, or `$(...)` in arguments,
bash will interpolate them before mesh-tool sees the value. Use stdin mode
with a single-quoted heredoc to avoid this:

```bash
mesh-tool gmail_send_message --to "someone@example.com" --subject "Invoice" --body - <<'EOF'
The payment of $550 is still outstanding.
EOF
```

The `--param -` syntax reads that parameter's value from stdin. The
single-quoted `<<'EOF'` prevents all shell expansion inside the heredoc.

### Error handling

- Missing or wrong arguments print usage and exit with code 1
- Successful calls return JSON to stdout with exit code 0
- Run `mesh-tool <name>` with no args to see required and optional parameters
