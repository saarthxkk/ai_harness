"""Anthropic-compatible tool definitions (schemas).

Each schema follows the Anthropic tool-use format:
    {
        "name": "<tool_name>",
        "description": "<what it does>",
        "input_schema": { ... JSON Schema ... }
    }

The ``TOOL_SCHEMAS`` list can be passed directly to ``call_model(tools=...)``.
``TOOL_DISPATCH`` maps tool names to their implementing callables so the
orchestrator can execute tool calls returned by the model.
"""

from __future__ import annotations

from typing import Any, Callable

from src.tools import file_ops, shell, search

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

LIST_FILES_SCHEMA: dict[str, Any] = {
    "name": "list_files",
    "description": (
        "List files and directories at the given path relative to the "
        "repository root. Returns names, relative paths, and whether each "
        "entry is a directory."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Directory path relative to the repository root. "
                    "Defaults to '.' (the root itself)."
                ),
            },
        },
        "required": [],
    },
}

READ_FILE_SCHEMA: dict[str, Any] = {
    "name": "read_file",
    "description": (
        "Read the UTF-8 contents of a file at the given path. "
        "The path must be relative to the repository root or an absolute "
        "path inside the repository."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the repository root.",
            },
        },
        "required": ["path"],
    },
}

WRITE_FILE_SCHEMA: dict[str, Any] = {
    "name": "write_file",
    "description": (
        "Overwrite an existing file with new content. The file must "
        "already exist. Use create_file to make new files."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the repository root.",
            },
            "content": {
                "type": "string",
                "description": "The full content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
}

CREATE_FILE_SCHEMA: dict[str, Any] = {
    "name": "create_file",
    "description": (
        "Create a new file with the given content. Parent directories "
        "are created automatically. Fails if the file already exists."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the repository root.",
            },
            "content": {
                "type": "string",
                "description": "Initial content for the file. Defaults to empty.",
            },
        },
        "required": ["path"],
    },
}

DELETE_FILE_SCHEMA: dict[str, Any] = {
    "name": "delete_file",
    "description": (
        "Delete a file inside the repository. This is a DESTRUCTIVE "
        "operation and is blocked by default policy."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the repository root.",
            },
        },
        "required": ["path"],
    },
}

FILE_EXISTS_SCHEMA: dict[str, Any] = {
    "name": "file_exists",
    "description": (
        "Check whether a file or directory exists at the given path "
        "inside the repository."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path relative to the repository root.",
            },
        },
        "required": ["path"],
    },
}

RUN_COMMAND_SCHEMA: dict[str, Any] = {
    "name": "run_command",
    "description": (
        "Execute a shell command from the repository root directory. "
        "Returns stdout, stderr, exit code, and duration. Destructive "
        "commands are blocked."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": (
                    "Maximum seconds to allow the command to run. "
                    "Defaults to 30, clamped to [1, 300]."
                ),
            },
        },
        "required": ["command"],
    },
}

TEXT_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "text_search",
    "description": (
        "Search file contents for a text string or regex pattern. "
        "Returns matching file paths, line numbers, content, and "
        "surrounding code snippets."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search string or regex pattern.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Subdirectory to search within, relative to repo root. "
                    "Defaults to '.' (entire repo)."
                ),
            },
            "glob_pattern": {
                "type": "string",
                "description": (
                    "File-name glob filter, e.g. '*.py'. "
                    "Defaults to '*' (all files)."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results. Defaults to 50.",
            },
            "use_regex": {
                "type": "boolean",
                "description": (
                    "Treat query as a regex pattern instead of a literal "
                    "string. Defaults to false."
                ),
            },
        },
        "required": ["query"],
    },
}

FILENAME_SEARCH_SCHEMA: dict[str, Any] = {
    "name": "filename_search",
    "description": (
        "Search for files and directories whose name matches a query "
        "string or regex. Returns relative paths."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Substring or regex to match against file names.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Subdirectory to search within, relative to repo root. "
                    "Defaults to '.' (entire repo)."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results. Defaults to 50.",
            },
            "use_regex": {
                "type": "boolean",
                "description": (
                    "Treat query as a regex pattern instead of a literal "
                    "string. Defaults to false."
                ),
            },
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# Aggregated lists
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    LIST_FILES_SCHEMA,
    READ_FILE_SCHEMA,
    WRITE_FILE_SCHEMA,
    CREATE_FILE_SCHEMA,
    DELETE_FILE_SCHEMA,
    FILE_EXISTS_SCHEMA,
    RUN_COMMAND_SCHEMA,
    TEXT_SEARCH_SCHEMA,
    FILENAME_SEARCH_SCHEMA,
]

# ---------------------------------------------------------------------------
# Dispatch table — maps tool name → callable
# ---------------------------------------------------------------------------


def _dispatch_list_files(path: str = ".") -> dict[str, Any]:
    return file_ops.list_files(path)


def _dispatch_read_file(path: str) -> dict[str, Any]:
    return file_ops.read_file(path)


def _dispatch_write_file(path: str, content: str) -> dict[str, Any]:
    return file_ops.write_file(path, content)


def _dispatch_create_file(path: str, content: str = "") -> dict[str, Any]:
    return file_ops.create_file(path, content)


def _dispatch_delete_file(path: str) -> dict[str, Any]:
    return file_ops.delete_file(path)


def _dispatch_file_exists(path: str) -> dict[str, Any]:
    return file_ops.file_exists(path)


def _dispatch_run_command(command: str, timeout: int = 30) -> dict[str, Any]:
    return shell.run_command(command, timeout=timeout)


def _dispatch_text_search(**kwargs: Any) -> dict[str, Any]:
    return search.text_search(**kwargs)


def _dispatch_filename_search(**kwargs: Any) -> dict[str, Any]:
    return search.filename_search(**kwargs)


TOOL_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "list_files": _dispatch_list_files,
    "read_file": _dispatch_read_file,
    "write_file": _dispatch_write_file,
    "create_file": _dispatch_create_file,
    "delete_file": _dispatch_delete_file,
    "file_exists": _dispatch_file_exists,
    "run_command": _dispatch_run_command,
    "text_search": _dispatch_text_search,
    "filename_search": _dispatch_filename_search,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Look up *name* in the dispatch table and call it with *arguments*.

    Returns a structured error dict if the tool name is unknown or
    execution raises.
    """
    handler = TOOL_DISPATCH.get(name)
    if handler is None:
        return {
            "success": False,
            "error": f"Unknown tool: {name!r}",
            "operation": name,
        }
    try:
        return handler(**arguments)
    except PermissionError as exc:
        return {
            "success": False,
            "error": str(exc),
            "operation": name,
            "risk": "BLOCKED",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "operation": name,
        }
