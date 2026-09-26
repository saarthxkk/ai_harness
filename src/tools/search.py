"""Tool — Code and file search utilities."""

import os
import re


def grep(pattern: str, directory: str, file_glob: str = "*") -> list[dict]:
    """Search for a regex pattern across files in a directory.

    Returns a list of matches with file path, line number, and line content.
    """
    matches: list[dict] = []
    regex = re.compile(pattern)

    for root, _dirs, files in os.walk(directory):
        for filename in files:
            if file_glob != "*" and not filename.endswith(file_glob):
                continue
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for lineno, line in enumerate(f, start=1):
                        if regex.search(line):
                            matches.append(
                                {
                                    "file": filepath,
                                    "line": lineno,
                                    "content": line.rstrip(),
                                }
                            )
            except (OSError, UnicodeDecodeError):
                continue

    return matches
