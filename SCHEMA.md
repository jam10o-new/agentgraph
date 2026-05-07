# Schema-Driven Output

AgentGraph supports llguidance-constrained inference via schema files placed in an agent's system directory. When a `.schema-*` file is present, the model's output is constrained to match the specified format, and output files use a custom naming pattern.

## Quick start

Create a file in your agent's system directory:

```bash
# agents/supervisor/system/.schema-rating.json
```

```json
{
  "type": "object",
  "properties": {
    "rating": { "type": "integer", "minimum": 1, "maximum": 5 },
    "reasoning": { "type": "string" }
  },
  "required": ["rating", "reasoning"]
}
```

The agent will now produce `rating_0.json`, `rating_1.json`, etc. — always valid JSON matching that schema.

## File naming

```
.schema-{format}.{extension}[@{constraint_type}]
```

| Part | Required | Description |
|------|----------|-------------|
| `format` | yes | Output filename pattern. Supports `{timestamp}` (epoch ms) and `{turn}` (auto-incremented count). |
| `extension` | yes | Output file extension. Also used as the default constraint type for `.json`. |
| `@{constraint_type}` | no | Overrides the inferred constraint. Use when the constraint type differs from the natural default. |

### Examples

| Schema file | Constraint | Output files |
|-------------|------------|--------------|
| `.schema-rating.json` | `JsonSchema` (default) | `rating_0.json`, `rating_1.json` |
| `.schema-out-{timestamp}.json` | `JsonSchema` | `out-1712345678000.json`, `out-1712345679000.json` |
| `.schema-log.txt@regex` | `Regex` | `log_0.txt`, `log_1.txt` |
| `.schema-story.txt@lark` | `Lark` | `story_0.txt` |
| `.schema-output.json@llg` | `Llguidance` | `output_0.json` |

## Constraint types

### `JsonSchema` (`.json` extension, or `@json`)

The file must contain a valid [JSON Schema](https://json-schema.org/) object. The model's output is constrained to produce JSON matching this schema. Uses mistralrs' `Constraint::JsonSchema` backed by llguidance.

```json
{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}}}
```

### `Regex` (`@regex`)

The file contains a regex pattern. The model's output must match this pattern.

```
[A-Z][a-z]+ the [A-Z][a-z]+
```

### `Lark` (`@lark`)

The file contains a [Lark grammar](https://lark-parser.readthedocs.io/). The model's output must conform to this grammar.

```
start: "yes" | "no"
```

### `Llguidance` (`@llg`)

The file contains a raw llguidance `TopLevelGrammar` JSON. For advanced use cases where you need full control over the constraint specification.

```json
{"grammars":[{"name":"main","json_schema":{"type":"object","properties":{"x":{"type":"integer"}}}}]}
```

## Format string variables

| Variable | Value |
|----------|-------|
| `{timestamp}` | Unix epoch milliseconds at inference time |
| `{turn}` | Zero-based counter — incremented for each existing file matching the pattern |

The turn counter is derived by scanning the output directory for existing files whose names match the format prefix and extension. This means restarting the agent resets the counter (empty output dir → `0` again).

## Tool interaction

When a schema is active, tool call appending to the output file is **disabled**. The constrained output is expected to be the complete response — appending raw tool call JSON would break the format. Tool calls are still executed normally during inference; they just don't appear in the output file.

## Downstream consumption

A distillation harness or downstream agent can:

1. **Detect schema usage**: Look for `.schema-*` files in the agent's system directory
2. **Find output files**: Match files in the output directory by the format pattern (prefix + extension)
3. **Validate output**: Parse output files against the schema — they're guaranteed to be valid by llguidance constrained decoding
4. **Reconstruct the inference request**: Use the schema content + system prompt to rebuild the exact prompt the model received

## Discovery from system directories

```python
import os, re, json

def discover_schema(system_dir):
    """Find and parse any .schema-* file in a system directory."""
    for f in os.listdir(system_dir):
        m = re.match(r'^\.schema-(.+)\.([a-z]+)(?:@([a-z]+))?$', f)
        if not m:
            continue
        format_str, ext, constraint_type = m.groups()
        constraint_type = constraint_type or ("json" if ext == "json" else None)
        if not constraint_type:
            continue
        with open(os.path.join(system_dir, f)) as fh:
            content = fh.read()
        return {
            "format": format_str,
            "extension": ext,
            "constraint_type": constraint_type,
            "content": content if constraint_type != "json" else json.loads(content),
        }
    return None
```
