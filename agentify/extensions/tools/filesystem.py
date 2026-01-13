import os
from typing import Any, Dict, List, Optional
from agentify.core.tool import Tool

DEFAULT_MAX_READ_BYTES = 1024 * 1024
HARD_MAX_READ_BYTES = 5 * 1024 * 1024

class BaseFilesystemTool(Tool):
    """Base class for filesystem tools with sandbox security."""
    
    def __init__(self, schema: Dict[str, Any], func: Any, sandbox_dir: Optional[str] = None):
        super().__init__(schema, func)
        # Default to current working directory if no sandbox is provided.
        self.sandbox_dir = os.path.abspath(sandbox_dir or os.getcwd())

    def _validate_path(self, file_path: str) -> str:
        """Ensure path is within sandbox."""
        # Resolve user path relative to sandbox.
        # Note: os.path.join discards sandbox_dir if file_path is absolute.
        full_path = os.path.join(self.sandbox_dir, file_path)
        
        # Resolve symlinks and .. components
        real_path = os.path.realpath(full_path)
        real_sandbox = os.path.realpath(self.sandbox_dir)

        # Check if the resolved path starts with the resolved sandbox path
        if os.path.commonpath([real_sandbox, real_path]) != real_sandbox:
             raise ValueError(f"Access denied: Path '{file_path}' resolves to '{real_path}', which is outside sandbox '{real_sandbox}'")
        
        return real_path


class ListDirTool(BaseFilesystemTool):
    def __init__(self, sandbox_dir: Optional[str] = None):
        schema = {
            "name": "list_files",
            "description": "List files and directories in a given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory_path": {
                        "type": "string",
                        "description": "Relative path to list contents of. Defaults to root of sandbox.",
                    }
                },
            },
        }
        super().__init__(schema, self._list_dir, sandbox_dir)

    def _list_dir(self, directory_path: str = ".") -> str:
        try:
            target_path = self._validate_path(directory_path)
            if not os.path.exists(target_path):
                return f"Error: Directory '{directory_path}' does not exist."
            
            items = os.listdir(target_path)
            # Add indicators for directories
            formatted_items = []
            for item in items:
                if os.path.isdir(os.path.join(target_path, item)):
                    formatted_items.append(f"{item}/")
                else:
                    formatted_items.append(item)
            
            return "\n".join(formatted_items) if formatted_items else "(empty directory)"
        except Exception as e:
            return f"Error listing directory: {str(e)}"


class ReadFileTool(BaseFilesystemTool):
    def __init__(self, sandbox_dir: Optional[str] = None):
        schema = {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "Maximum bytes to read from the file (hard-capped).",
                    },
                },
                "required": ["file_path"],
            },
        }
        super().__init__(schema, self._read_file, sandbox_dir)

    def _read_file(self, file_path: str, max_bytes: Optional[int] = None) -> str:
        try:
            target_path = self._validate_path(file_path)
            if not os.path.exists(target_path):
                return f"Error: File '{file_path}' does not exist."

            if max_bytes is not None and max_bytes <= 0:
                return "Error: 'max_bytes' must be a positive integer."

            read_limit = min(max_bytes or DEFAULT_MAX_READ_BYTES, HARD_MAX_READ_BYTES)

            with open(target_path, "rb") as f:
                content = f.read(read_limit + 1)

            truncated = len(content) > read_limit
            text = content[:read_limit].decode("utf-8", errors="replace")

            if truncated:
                return f"{text}\n[Truncated to {read_limit} bytes]"

            return text
        except Exception as e:
            return f"Error reading file: {str(e)}"


class WriteFileTool(BaseFilesystemTool):
    def __init__(self, sandbox_dir: Optional[str] = None):
        schema = {
            "name": "write_file",
            "description": "Write content to a file. Overwrites if exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file.",
                    }
                },
                "required": ["file_path", "content"],
            },
        }
        super().__init__(schema, self._write_file, sandbox_dir)

    def _write_file(self, file_path: str, content: str) -> str:
        try:
            target_path = self._validate_path(file_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            return f"Successfully wrote to '{file_path}'."
        except Exception as e:
            return f"Error writing file: {str(e)}"
