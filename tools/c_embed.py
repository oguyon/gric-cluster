#!/usr/bin/env python3
"""
Common C string embedding and escaping utilities for code generators.
"""

def escape_c_string(text, indent=4, chunk_size=65):
    """Escape text for inclusion in C string literal with line wrapping."""
    lines = text.split("\n")
    escaped_lines = []
    pad = " " * indent
    for line in lines:
        for i in range(0, max(len(line), 1), chunk_size):
            chunk = line[i:i + chunk_size]
            if i + chunk_size >= len(line):
                chunk += "\n"
            esc = chunk.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
            escaped_lines.append(f'{pad}"{esc}"')
    return "\n".join(escaped_lines)
