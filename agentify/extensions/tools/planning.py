from typing import Any, Dict, List, Optional
from agentify.core.tool import Tool

class TodoTool(Tool):
    """A tool for agents to manage their own todo list / plan."""
    
    def __init__(self):
        self._todos: List[Dict[str, Any]] = []
        schema = {
            "name": "manage_plan",
            "description": "Manage a todo list to plan and track progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "complete", "list", "remove"],
                        "description": "Action to perform on the plan.",
                    },
                    "task": {
                        "type": "string",
                        "description": "Description of the task (required for 'add')",
                    },
                    "task_id": {
                        "type": "integer",
                        "description": "ID of the task (required for 'complete' or 'remove')",
                    },
                },
                "required": ["action"],
            },
        }
        super().__init__(schema, self._manage_todos)

    def _manage_todos(
        self, 
        action: str, 
        task: Optional[str] = None, 
        task_id: Optional[int] = None
    ) -> str:
        if action == "add":
            if not task:
                return "Error: 'task' description required for 'add' action."
            new_id = len(self._todos)
            self._todos.append({"id": new_id, "task": task, "status": "pending"})
            return f"Task added: [{new_id}] {task}"
        
        elif action == "complete":
            if task_id is None:
                return "Error: 'task_id' required for 'complete' action."
            if 0 <= task_id < len(self._todos):
                self._todos[task_id]["status"] = "completed"
                return f"Task [{task_id}] marked as completed."
            return f"Error: Invalid task_id {task_id}"
            
        elif action == "remove":
            if task_id is None:
                return "Error: 'task_id' required for 'remove' action."
            if 0 <= task_id < len(self._todos):
                # Rebuild list without the specified task_id
                self._todos = [t for i, t in enumerate(self._todos) if i != task_id]
                # Re-assign IDs to keep them sequential
                for i, t in enumerate(self._todos):
                    t["id"] = i
                return f"Task removed. Remaining tasks re-indexed."
            return f"Error: Invalid task_id {task_id}"

        elif action == "list":
            if not self._todos:
                return "Plan is empty."
            lines = []
            for t in self._todos:
                status = "[x]" if t["status"] == "completed" else "[ ]"
                lines.append(f"{status} {t['id']}: {t['task']}")
            return "\n".join(lines)
        
        return f"Error: Unknown action '{action}'"
