import json


def json_parser(file: str) -> dict | None:
    try:
        file = file.replace("```", "").replace("json", "").strip()
        return json.loads(file)
    except json.JSONDecodeError:
        return None
