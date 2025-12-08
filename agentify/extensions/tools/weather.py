from agentify.core.tool import Tool
import os
import requests
from dotenv import load_dotenv

load_dotenv()


class WeatherTool(Tool):
    """Tool for getting current weather information."""

    def __init__(self):
        schema = {
            "name": "get_weather",
            "description": "Obtiene el estado del tiempo o clima actual para una ciudad o zona especificada.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Nombre de la ciudad o zona para consultar el clima.",
                    }
                },
                "required": ["location"],
            },
        }
        super().__init__(schema, self._get_weather)

    def _get_weather(self, location: str):
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return {"error": "Variable de entorno OPENWEATHER_API_KEY no configurada."}
        try:
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={"q": location, "appid": api_key, "units": "metric"},
            )
            data = response.json()
            if response.status_code != 200:
                return {
                    "error": data.get("message", "Error desconocido al obtener el clima.")
                }
            weather = {
                "location": data["name"],
                "description": data["weather"][0]["description"],
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "wind_speed": data["wind"]["speed"],
            }
            return {"weather": weather}
        except Exception as e:
            return {"error": f"Error al conectar con el servicio de clima: {e}"}
