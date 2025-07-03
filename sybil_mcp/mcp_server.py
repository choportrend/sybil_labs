from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import importlib
import os

app = FastAPI(title="Sybil MCP Server")

# Plugin system
PLUGINS = {}
PLUGINS_DIR = os.path.join(os.path.dirname(__file__), "plugins")

# Auto-discover and load plugins
for fname in os.listdir(PLUGINS_DIR):
    if fname.endswith(".py") and fname != "__init__.py":
        mod_name = f"plugins.{fname[:-3]}"
        mod = importlib.import_module(mod_name)
        if hasattr(mod, "register_plugin"):
            plugin = mod.register_plugin()
            PLUGINS[plugin.name] = plugin

# Image generation endpoint
class ImageRequest(BaseModel):
    prompt: str
    user_id: str = None  # Optional, for future per-user context

@app.post("/generate_image")
def generate_image(req: ImageRequest):
    plugin = PLUGINS.get("image_gen")
    if not plugin:
        raise HTTPException(status_code=500, detail="Image generation plugin not loaded.")
    try:
        result = plugin.generate_image(req.prompt, user_id=req.user_id)
        return {"result": result}
    except Exception as e:
        import traceback
        traceback.print_exc()  # Print the full error traceback for debugging
        raise HTTPException(status_code=500, detail=str(e)) 