# Telegram HTML Render — f-string Pitfall

## The Problem

Telegram's `parse_mode=HTML` supports `<sub>` and `<sup>` tags, but nested `{}` inside f-strings are **always evaluated by Python before the string is sent to Telegram**.

If you write in an f-string:
```python
f"GARCH: σ²_t = ω + αε²_{t-1} + βσ²_{t-1}"
```
and `t=564` at the end of a loop, Telegram receives `σ²_564` instead of `σ²_{t-1}`.

## The Fix — Double Braces

Use `{{}}` to escape literal braces in f-strings:
```python
f"GARCH: σ²_t = ω + αε²_{{t-1}} + βσ²_{{t-1}}"
# Telegram receives: σ²_{t-1} ✓
```

Same for any literal brace content:
```python
f"HAR: log(RV_t) = α + β₁log(RV_{{t-1}}) + ..."
# Telegram receives: log(RV_{t-1}) ✓
```

## Root Cause

Python's f-string evaluation happens first (server-side), then the result is sent to Telegram. Telegram never sees the template — only the evaluated output.

## Other Characters to Watch

| Content | f-string写法 | Telegram收到 |
|---------|------------|------------|
| `{t-1}` | `{{t-1}}` | `{t-1}` ✓ |
| `{t}` | `{{t}}` | `{t}` ✓ |
| `%` (Telegram reserved) | `{{"}}` | `"` ✓ |
| `<` `>` (HTML chars) | Use `&lt;` `&gt;` in HTML mode | HTML renders correctly |

### `%` is reserved in some contexts
Avoid bare `%` in HTML mode strings, or escape with `%%` in f-strings.

### `<br>` not supported — use `\n` instead
Telegram's HTML parser does **not** support the `<br>` tag:
```
Bad Request: can't parse entities: Unsupported start tag "br" at byte offset 33
```
Use `\n` (newline) instead of `<br>` to join multi-line messages. Telegram renders newlines correctly in HTML mode.

### HTML-reserved characters must be escaped
In HTML mode, `<`, `>`, `&` must be written as `&lt;`, `&gt;`, `&amp;`.

## Checklist

- [ ] Any mathematical subscript in Telegram HTML report → `{{...}}`
- [ ] Any literal `{}` in Telegram HTML report → `{{...}}`
- [ ] Multi-line messages → use `\n`, NOT `<br>`
- [ ] HTML characters (`<`, `>`, `&`) → escape as `&lt;`, `&gt;`, `&amp;`
- [ ] `%` in HTML strings → avoid or double-escape
