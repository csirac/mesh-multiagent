## Tool Calling Syntax

You have access to tools that let you interact with external systems. To call a tool, use this exact XML format:

```xml
<mesh_call name="tool_name">
<param_name>value</param_name>
<another_param>value</another_param>
</mesh_call>
```

### Rules

1. **Tool name** goes in the `name` attribute: `<mesh_call name="tool_name">`
2. **Parameters** are child elements: `<param_name>value</param_name>`
3. **Case-sensitive** – parameter names must match exactly
4. **JSON values** – for objects/arrays, include valid JSON inside the parameter
5. **Results** are returned in `<mesh_result name="tool_name">` elements

### Examples

**Calling a tool:**
```xml
<mesh_call name="gmail_list_from_date">
<date>2026-01-22</date>
</mesh_call>
```

**Discovering nodes:**
```xml
<mesh_call name="mesh_list"></mesh_call>
```

**Listing channels:**
```xml
<mesh_call name="channel_list"></mesh_call>
```

**Checking channel members:**
```xml
<mesh_call name="channel_members">
<channel_name>research</channel_name>
</mesh_call>
```

**Using sleep to not respond:**
```xml
<mesh_call name="sleep">
<reason>Agent-only status update in channel; no user-visible reply needed.</reason>
</mesh_call>
```

**Getting tool help:**
```xml
<mesh_call name="tool_help">
<tool_name>gmail_send_message</tool_name>
</mesh_call>
```
