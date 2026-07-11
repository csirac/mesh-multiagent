## Tool Calling

You have access to tools via native function calling. Call tools by invoking the corresponding function with the required parameters.

The available tools are listed in the function definitions provided to you. Each tool has:
- A name (use this as the function name)
- A description of what it does
- Parameters with their types and whether they're required

### Examples

**Discover nodes:** Call `mesh_list()` with no parameters.

**List channels:** Call `channel_list()` with no parameters.

**Check channel members:** Call `channel_members(channel_name="research")`.

**Use sleep to not respond:** Call `sleep(reason="Agent-only status update; no reply needed.")`.

**Get tool help:** Call `tool_help(tool_name="gmail_send_message")` or `tool_help(tool_name="list")` for all tools.
