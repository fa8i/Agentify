from agentify.core.tool import Tool
import datetime


get_current_time_schema = {
    "name": "get_current_time",
    "description": "Devuelve la hora y fecha actual en formato ISO 8601.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def get_current_time():
    now = datetime.datetime.now().astimezone().isoformat()
    return {"current_time": now}


get_current_time_tool = Tool(get_current_time_schema, get_current_time)
