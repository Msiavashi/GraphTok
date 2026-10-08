"""Line-oriented formatting for committed benchmark result JSON."""

from __future__ import annotations

import json


def _render(value: object, level: int) -> list[str]:
    indentation = "  " * level
    child_indentation = "  " * (level + 1)

    if isinstance(value, dict):
        if not value:
            return [indentation + "{}"]

        lines = [indentation + "{"]
        entries = list(value.items())
        for index, (key, entry_value) in enumerate(entries):
            if not isinstance(key, str):
                raise TypeError("benchmark result object keys must be strings")
            rendered_value = _render(entry_value, level + 1)
            rendered_value[0] = (
                f"{child_indentation}{json.dumps(key)}: "
                f"{rendered_value[0][len(child_indentation):]}"
            )
            if index < len(entries) - 1:
                rendered_value[-1] += ","
            lines.extend(rendered_value)
        lines.append(indentation + "}")
        return lines

    if isinstance(value, list):
        if not value or all(not isinstance(item, (dict, list)) for item in value):
            return [indentation + json.dumps(value, separators=(",", ":"))]

        lines = [indentation + "["]
        for index, item in enumerate(value):
            suffix = "," if index < len(value) - 1 else ""
            lines.append(
                child_indentation
                + json.dumps(item, separators=(",", ":"))
                + suffix
            )
        lines.append(indentation + "]")
        return lines

    return [indentation + json.dumps(value)]


def format_result_json(value: object) -> str:
    """Keep objects readable while storing arrays as one item per line."""
    return "\n".join(_render(value, 0))
