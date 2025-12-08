import os
from typing import Any, Dict, List, Optional
from agentify.core.tool import Tool

class BaseFilesystemTool(Tool):
    """Base class for filesystem tools with sandbox security."""
    
    def __init__(self, schema: Dict[str, Any], func: Any, sandbox_dir: Optional[str] = None):
        super().__init__(schema, func)
        # If no sandbox provided, default to current working directory or a safe temp dir could be better 
        # but for this agent library, let's default to CWD but allow override.
        self.sandbox_dir = os.path.abspath(sandbox_dir or os.getcwd())

    def _validate_path(self, file_path: str) -> str:
        """Ensure path is within sandbox."""
        # Handle absolute paths by checking if they start with sandbox
        abs_path = os.path.abspath(os.path.join(self.sandbox_dir, file_path))
        
        if not abs_path.startswith(self.sandbox_dir):
            raise ValueError(f"Access denied: Path '{file_path}' is outside sandbox directory '{self.sandbox_dir}'")
        
        return abs_path


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
                    }
                },
                "required": ["file_path"],
            },
        }
        super().__init__(schema, self._read_file, sandbox_dir)

    def _read_file(self, file_path: str) -> str:
        try:
            target_path = self._validate_path(file_path)
            if not os.path.exists(target_path):
                return f"Error: File '{file_path}' does not exist."
            
            with open(target_path, "r", encoding="utf-8") as f:
                return f.read()
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
