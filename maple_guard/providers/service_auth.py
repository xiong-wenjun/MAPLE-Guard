"""Resolve service credentials without putting them in CLI arguments or traces."""
import json
import os
from pathlib import Path
def service_headers(base_url, environment_keys):
    credentials=os.getenv("MAPLE_SERVICE_CREDENTIALS","")
    if credentials:
        services=json.loads(Path(credentials).read_text())
        for service in services.values():
            if service.get("base_url","").rstrip("/")==base_url.rstrip("/"):
                key=service.get("api_key","")
                return {"Authorization":"Bearer "+key} if key else {}
        raise ValueError("Endpoint absent from configured service credential file")
    key=next((os.getenv(name) for name in environment_keys if os.getenv(name)), "")
    return {"Authorization":"Bearer "+key} if key else {}

def chat_headers(base_url):
    return service_headers(base_url, ("CHAT_API_KEY", "OPENAI_API_KEY"))

def embedding_headers(base_url):
    return service_headers(base_url, ("EMBED_API_KEY",))
