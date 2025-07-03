# Sybil MCP Server

This is the Model Context Protocol (MCP) server for Sybil Labs. It provides a modular, plugin-based API for AI assistant features, starting with image generation via OpenAI DALL·E.

## Features
- Modular plugin system (first plugin: image generation)
- FastAPI-based HTTP API
- Ready for local or cloud deployment

## Getting Started

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Run the server:**
   ```sh
   uvicorn mcp_server:app --reload
   ```

## Plugins
- Plugins live in the `plugins/` directory and implement a simple interface.
- The server auto-discovers and loads plugins at startup.

## Example Request
```
POST /generate_image
{
  "prompt": "A cat in a spacesuit"
}
```

## License
MIT 