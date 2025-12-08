from agentify.core.tool import Tool
import datetime


class TimeTool(Tool):
    """Tool for getting current date and time."""

    def __init__(self):
        schema = {
            "name": "get_current_time",
            "description": "Devuelve la hora y fecha actual en formato ISO 8601.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
        super().__init__(schema, self._get_current_time)

    def _get_current_time(self):
        now = datetime.datetime.now().astimezone().isoformat()
        return {"current_time": now}
