from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import importlib
import os
import openai
import re
import tiktoken
from fastapi.responses import JSONResponse
import logging

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

# In-memory per-user context and model
USER_CONTEXT: Dict[str, List[Dict[str, str]]] = {}
USER_MODEL: Dict[str, str] = {}
SUPPORTED_MODELS = [
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo"
]
DEFAULT_MODEL = "gpt-4o"

SYSTEM_PROMPT = (
    "You are Sybil, an AI-powered virtual assistant. "
    "You help the user (your owner) with a variety of tasks, including answering questions, managing to-do lists, and more. "
    "You may be extended in the future with additional tools and MCP clients to access real-time data, perform web searches, manage calendars, and more. "
    "Always be helpful, proactive, and concise. "
    "If the user requests an image (e.g., says 'draw', 'generate an image', 'show me a picture', etc.), reply conversationally as usual, and then on a new line, add [image: <a detailed, creative prompt for an AI image generator, based on the user's request and the conversation>]. "
    "Invent a vivid, specific, and imaginative prompt as if you were a professional AI artist or prompt engineer. "
    "For example, if the user says 'draw a cat', you might write: [image: A photorealistic cat wearing a wizard hat, sitting on a stack of ancient books, in a candle-lit library]. "
    "If the user says 'draw an image that represents you as an AI assistant', you might write: [image: A glowing, futuristic digital landscape with interconnected data streams, holographic displays, and a friendly AI avatar at the center, surrounded by icons of tasks and communication]. "
    "Do not include any other text on the same line as the [image: ...] tag. "
    "CRITICAL: For ALL to-do list actions, you MUST ALWAYS use a [todo: ...] tag. This is MANDATORY and NON-NEGOTIABLE. "
    "Examples of REQUIRED tags for each action: "
    "- To add a task: [todo: add buy milk] "
    "- To mark a task as done: [todo: done 2] (where 2 is the task number) "
    "- To mark a task as not done: [todo: undone 2] "
    "- To remove a task: [todo: remove 3] "
    "- To show the list: [todo: list] "
    "Even if the user asks in natural language like 'mark task 5 as done' or 'remove the third item', you MUST output the corresponding [todo: ...] tag. "
    "If you do not use a [todo: ...] tag, the user's list will NOT be updated. Never confirm an action unless you have output the correct tag. "
    "ALWAYS use emoji (✅ for done, ❌ for not done) for to-do list formatting. "
    "ALWAYS confirm the action in your reply, e.g. 'Added to your to-do list: buy milk.' or 'Here is your to-do list: ...' or 'Marked task 2 as done.' or 'Removed task 1.' "
    "For all other requests, respond normally. "
    "If you need up-to-date information from the internet, or the user asks a question that requires a web search, output a [web: ...] tag on a new line with the search query. "
    "For example, if the user says 'What's the latest news about AI?', you might write: [web: latest news about AI]. "
    "After the web result is provided, use it to answer the user's question naturally, as if you just knew the information."
    "For all note actions, you MUST ALWAYS use a [note: ...] tag. This is MANDATORY and NON-NEGOTIABLE. "
    "Examples of REQUIRED tags for each action: "
    "- To add a note: [note: add Call Alice about the contract] "
    "- To list notes: [note: list] "
    "- To search notes: [note: search Project X] "
    "- To show a note: [note: show 2] (where 2 is the note number) "
    "- To remove a note: [note: remove 3] "
    "Even if the user asks in natural language like 'take a note', 'show my notes', or 'remove the third note', you MUST output the corresponding [note: ...] tag. "
    "If you do not use a [note: ...] tag, the user's notes will NOT be updated. Never confirm an action unless you have output the correct tag. "
    "ALWAYS confirm the action in your reply, e.g. 'Noted: ...', 'Here are your notes: ...', 'Removed note 1.', etc. "
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    text: Optional[str] = None
    image_url: str = ""
    image_prompt: Optional[str] = None

class ModelRequest(BaseModel):
    user_id: str
    model: str

class UserRequest(BaseModel):
    user_id: str

class TodoRequest(BaseModel):
    user_id: str
    task: str = None
    index: int = None
    query: str = None

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # Manage per-user context
    history = USER_CONTEXT.setdefault(req.user_id, [])
    history.append({"role": "user", "content": req.message})
    # Limit context to last 50 messages
    history = history[-50:]
    USER_CONTEXT[req.user_id] = history
    # Add system prompt at the start
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Inject user's to-do list as a system message if non-empty, using emoji
    todo_plugin = PLUGINS.get("todo")
    if todo_plugin:
        todos = todo_plugin.get_todo_list(req.user_id)
        if todos:
            todo_lines = [f"{i+1}. {'✅' if t['done'] else '❌'} {t['task']}" for i, t in enumerate(todos)]
            todo_text = "Current to-do list for this user:\n" + "\n".join(todo_lines)
            message_history.append({"role": "system", "content": todo_text})
    message_history += history
    # Use per-user model if set
    model = USER_MODEL.get(req.user_id, DEFAULT_MODEL)
    if model not in SUPPORTED_MODELS:
        help_text = (
            f"The model '{model}' is not supported.\n"
            f"Available models: {', '.join(SUPPORTED_MODELS)}\n"
            "Use /model <model_name> to change."
        )
        return {"text": help_text, "image_url": "", "image_prompt": None}
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=message_history
        )
        llm_reply = response["choices"][0]["message"]["content"]
    except Exception as e:
        help_text = (
            f"Error communicating with LLM: {e}\n"
            f"Current model: {model}\n"
            f"Available models: {', '.join(SUPPORTED_MODELS)}\n"
            "Use /model <model_name> to change."
        )
        return {"text": help_text, "image_url": "", "image_prompt": None}
    # Debug logging: print LLM output and parsed tool tags
    print("\n--- MCP DEBUG ---")
    print(f"User: {req.user_id}")
    print(f"LLM raw output:\n{llm_reply}")
    todo_tags = re.findall(r"\[todo:\s*(.+?)\]", llm_reply, re.IGNORECASE | re.DOTALL)
    note_tags = re.findall(r"\[note:\s*(.+?)\]", llm_reply, re.IGNORECASE | re.DOTALL)
    print(f"Parsed [todo: ...] tags: {todo_tags}")
    print(f"Parsed [note: ...] tags: {note_tags}")
    # Web search tool logic
    web_plugin = PLUGINS.get("web_search")
    web_tags = re.findall(r"\[web:\s*(.+?)\]", llm_reply, re.IGNORECASE | re.DOTALL)
    if web_plugin and web_tags:
        web_contexts = []
        for web_query in web_tags:
            web_query = web_query.strip()
            print(f"[WEB DEBUG] Sybil is searching the web for: {web_query}")
            web_result = web_plugin.search_web(web_query, user_id=req.user_id)
            print(f"[WEB DEBUG] Web result: {web_result}")
            web_contexts.append({"role": "system", "content": f"Web search result for '{web_query}': {web_result}\nYou should reference the source(s) above in your answer if relevant."})
        # Remove all [web: ...] tags from the LLM reply for the next round
        llm_reply_clean = re.sub(r"^\[web: .+?\]$", "", llm_reply, flags=re.IGNORECASE | re.MULTILINE).strip()
        # Add all web contexts and the cleaned LLM reply to the message history
        message_history.extend(web_contexts)
        message_history.append({"role": "assistant", "content": llm_reply_clean})
        # Re-call the LLM so it can use all web results
        try:
            response2 = openai.ChatCompletion.create(
                model=model,
                messages=message_history
            )
            llm_reply2 = response2["choices"][0]["message"]["content"]
        except Exception as e:
            return {"text": f"Error after web search: {e}", "image_url": "", "image_prompt": None}
        # Store assistant response in context
        history.append({"role": "assistant", "content": llm_reply2})
        USER_CONTEXT[req.user_id] = history
        return {"text": llm_reply2, "image_url": "", "image_prompt": None}
    print("--- END DEBUG ---\n")
    # Tool triggers (robust parsing)
    todo_plugin = PLUGINS.get("todo")
    notes_plugin = PLUGINS.get("notes")
    todo_confirmation = ""
    note_confirmation = ""
    tool_triggered = False
    action_words = ["add", "added", "remove", "removed", "mark", "marked", "done", "complete", "completed", "unmark", "unmarked", "not done"]
    undone_fallback_triggered = False
    if todo_plugin:
        # Find all [todo: ...] tags (robust, multiline, extra spaces, case-insensitive)
        todo_tags = re.findall(r"\[todo:\s*(.+?)\]", llm_reply, re.IGNORECASE | re.DOTALL)
        # Sort remove tags descending by index, process them first
        remove_tags = []
        other_tags = []
        for tag in todo_tags:
            if tag.lower().startswith("remove "):
                remove_tags.append(tag)
            else:
                other_tags.append(tag)
        def extract_index(tag):
            try:
                return int(tag.split(" ", 1)[1])
            except Exception:
                return -1
        remove_tags.sort(key=extract_index, reverse=True)
        # Now process remove_tags first, then other_tags
        all_tags = remove_tags + other_tags
        for tag in all_tags:
            tag = tag.strip()
            if tag.lower().startswith("add "):
                result = todo_plugin.handle_todo("add", req.user_id, tag[4:].strip())
                print(f"[TODO DEBUG] add: {tag[4:].strip()} | result: {result}")
                todo_confirmation += f"\n{result}"
                tool_triggered = True
            elif tag.lower() == "list":
                result = todo_plugin.handle_todo("list", req.user_id)
                print(f"[TODO DEBUG] list | result: {result}")
                todo_confirmation += f"\n{result}"
                tool_triggered = True
            elif tag.lower().startswith("done "):
                result = todo_plugin.handle_todo("done", req.user_id, tag[5:].strip())
                print(f"[TODO DEBUG] done: {tag[5:].strip()} | result: {result}")
                todo_confirmation += f"\n{result}"
                tool_triggered = True
            elif tag.lower().startswith("remove "):
                result = todo_plugin.handle_todo("remove", req.user_id, tag[7:].strip())
                print(f"[TODO DEBUG] remove: {tag[7:].strip()} | result: {result}")
                todo_confirmation += f"\n{result}"
                tool_triggered = True
            elif tag.lower().startswith("undone "):
                result = todo_plugin.handle_todo("undone", req.user_id, tag[7:].strip())
                print(f"[TODO DEBUG] undone: {tag[7:].strip()} | result: {result}")
                todo_confirmation += f"\n{result}"
                tool_triggered = True
            # After each action, refresh the list for next tag
            if "Ambiguous task description" in result or "not found" in result or "couldn't find" in result:
                # Stop processing further tags if error
                print(f"[TODO DEBUG] Error encountered, stopping further tag processing: {result}")
                break
        # Enhanced fallback logic: if the LLM says action words but no [todo: ...] tag is present, try to infer and trigger the action
        if not tool_triggered:
            # Fallback for "mark as done" actions
            done_match = re.search(r"mark(?:\s+task)?\s+(\d+)\s+as\s+done|marked(?:\s+task)?\s+(\d+)\s+as\s+done|task\s+(\d+)\s+as\s+done", llm_reply, re.IGNORECASE)
            if done_match:
                idx = done_match.group(1) or done_match.group(2) or done_match.group(3)
                if idx:
                    result = todo_plugin.handle_todo("done", req.user_id, idx.strip())
                    todo_confirmation += f"\n{result}"
                    tool_triggered = True
                    print(f"FALLBACK TRIGGERED: done {idx}")
            
            # Fallback for "mark as not done" or "unmark as done" actions
            undone_match = re.search(r"mark(?:\s+task)?\s+(\d+)\s+as\s+not done|unmark(?:\s+task)?\s+(\d+)?\s*as\s*done|marked(?:\s+task)?\s+(\d+)\s+as\s+not done", llm_reply, re.IGNORECASE)
            if undone_match:
                idx = undone_match.group(1) or undone_match.group(2) or undone_match.group(3)
                if idx:
                    result = todo_plugin.handle_todo("undone", req.user_id, idx.strip())
                    todo_confirmation += f"\n{result}"
                    tool_triggered = True
                    undone_fallback_triggered = True
                    print(f"FALLBACK TRIGGERED: undone {idx}")
    # Notes tool triggers
    if notes_plugin:
        for tag in note_tags:
            tag = tag.strip()
            if tag.lower().startswith("add "):
                result = notes_plugin.handle_note("add", req.user_id, tag[4:].strip())
                print(f"[NOTES DEBUG] add: {tag[4:].strip()} | result: {result}")
                note_confirmation += f"\n{result}"
                tool_triggered = True
            elif tag.lower() == "list":
                result = notes_plugin.handle_note("list", req.user_id)
                print(f"[NOTES DEBUG] list | result: {result}")
                note_confirmation += f"\n{result}"
                tool_triggered = True
            elif tag.lower().startswith("search "):
                result = notes_plugin.handle_note("search", req.user_id, tag[7:].strip())
                print(f"[NOTES DEBUG] search: {tag[7:].strip()} | result: {result}")
                note_confirmation += f"\n{result}"
                tool_triggered = True
            elif tag.lower().startswith("show "):
                result = notes_plugin.handle_note("show", req.user_id, tag[5:].strip())
                print(f"[NOTES DEBUG] show: {tag[5:].strip()} | result: {result}")
                note_confirmation += f"\n{result}"
                tool_triggered = True
            elif tag.lower().startswith("remove "):
                result = notes_plugin.handle_note("remove", req.user_id, tag[7:].strip())
                print(f"[NOTES DEBUG] remove: {tag[7:].strip()} | result: {result}")
                note_confirmation += f"\n{result}"
                tool_triggered = True
    # Remove all [todo: ...] and [note: ...] tags from the reply
    text_reply = re.sub(r"^\[todo: .+?\]$", "", llm_reply, flags=re.IGNORECASE | re.MULTILINE).strip()
    text_reply = re.sub(r"^\[note: .+?\]$", "", text_reply, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Helper: does LLM reply already confirm the action?
    def llm_confirms_action(action_type, idx_or_task=None):
        # Lowercase for matching
        t = text_reply.lower()
        if action_type == "done":
            if idx_or_task:
                return (
                    f"marked task {idx_or_task} as done" in t or
                    f"task {idx_or_task} marked as done" in t or
                    f"✅" in t or
                    f"completed task {idx_or_task}" in t or
                    f"task {idx_or_task} completed" in t
                )
            return "marked as done" in t or "completed" in t or "✅" in t
        if action_type == "undone":
            if idx_or_task:
                return (
                    f"marked task {idx_or_task} as not done" in t or
                    f"task {idx_or_task} marked as not done" in t or
                    f"unmarked task {idx_or_task} as done" in t or
                    f"not done" in t
                )
            return "marked as not done" in t or "unmarked as done" in t or "not done" in t
        if action_type == "remove":
            if idx_or_task:
                return (
                    f"removed task {idx_or_task}" in t or
                    f"task {idx_or_task} removed" in t or
                    f"deleted task {idx_or_task}" in t or
                    f"removed: " in t or
                    f"deleted: " in t
                )
            return "removed" in t or "deleted" in t
        if action_type == "add":
            if idx_or_task:
                return (
                    f"added to your to-do list: {idx_or_task}" in t or
                    f"added: {idx_or_task}" in t or
                    f"added {idx_or_task}" in t
                )
            return "added to your to-do list" in t or "added" in t
        return False

    # Only append plugin confirmation if LLM did not already confirm
    # Try to detect which action was performed
    suppress_confirmation = False
    if tool_triggered and todo_tags:
        for tag in todo_tags:
            tag = tag.strip()
            if tag.lower().startswith("add "):
                task = tag[4:].strip()
                if llm_confirms_action("add", task):
                    suppress_confirmation = True
            elif tag.lower().startswith("done "):
                idx = tag[5:].strip()
                if llm_confirms_action("done", idx):
                    suppress_confirmation = True
            elif tag.lower().startswith("remove "):
                idx = tag[7:].strip()
                if llm_confirms_action("remove", idx):
                    suppress_confirmation = True
            elif tag.lower().startswith("undone "):
                idx = tag[7:].strip()
                if llm_confirms_action("undone", idx):
                    suppress_confirmation = True
    # Fallbacks
    if tool_triggered and not todo_tags:
        # Try to infer from fallback action
        if 'FALLBACK TRIGGERED: done' in todo_confirmation:
            m = re.search(r'done (\d+)', todo_confirmation)
            if m and llm_confirms_action("done", m.group(1)):
                suppress_confirmation = True
        if 'FALLBACK TRIGGERED: remove' in todo_confirmation:
            m = re.search(r'remove (\d+)', todo_confirmation)
            if m and llm_confirms_action("remove", m.group(1)):
                suppress_confirmation = True
        if 'FALLBACK TRIGGERED: undone' in todo_confirmation:
            m = re.search(r'undone (\d+)', todo_confirmation)
            if m and llm_confirms_action("undone", m.group(1)):
                suppress_confirmation = True
    # Refined warning logic: only show warning if the LLM's reply contains confirmation/action language, not just a list
    lower_reply = text_reply.lower()
    is_action = any(word in lower_reply for word in action_words)
    is_list_display = ("to-do list" in lower_reply or "todo list" in lower_reply or "here is your" in lower_reply or "here's your" in lower_reply)
    # Only show the to-do warning if the LLM reply is about the to-do list
    is_todo_context = (
        "to-do list" in lower_reply or "todo list" in lower_reply or
        "task" in lower_reply or "tasks" in lower_reply
    )
    if is_action and not tool_triggered and is_todo_context:
        todo_confirmation += "\n⚠️ Sorry, I couldn't update your to-do list because the action wasn't properly triggered. Please try again or use the /todo menu."
    # Only overwrite with the raw to-do list if the user's message is an explicit request
    explicit_show_todo = any(kw in req.message.lower() for kw in ["show my todo list", "show me my todo list", "what's on my todo list", "list my todos", "list my to-dos", "show todo list", "show to-do list"])
    if explicit_show_todo:
        todos = todo_plugin.get_todo_list(req.user_id)
        if todos:
            todo_lines = [f"{i+1}. {'✅' if t['done'] else '❌'} {t['task']}" for i, t in enumerate(todos)]
            text_reply = "Here's your to-do list:\n" + "\n".join(todo_lines)
        else:
            text_reply = "Your to-do list is empty."
        return {"text": text_reply, "image_url": "", "image_prompt": None}
    # Check for [image: ...] in the LLM response
    image_match = re.search(r"\[image: (.+?)\]", llm_reply, re.IGNORECASE | re.DOTALL)
    image_url = ""
    image_prompt = None
    # Fallback: detect image intent and creative description if no [image: ...] tag
    user_message = req.message.lower()
    image_intent_words = ["draw", "depict", "visualize", "picture", "illustrate", "show me a picture", "make me an image", "generate an image"]
    creative_starts = ["imagine", "picture", "visualize", "envision", "a scene", "a depiction", "a creative representation", "a surreal", "a whimsical", "a vibrant", "a collage", "an illustration"]
    def has_image_intent(msg):
        msg = msg.lower()
        return any(word in msg for word in image_intent_words)
    def extract_creative_description(reply):
        # Look for a paragraph or sentence that starts with a creative word
        for line in reply.split("\n"):
            l = line.strip().lower()
            if any(l.startswith(start) for start in creative_starts):
                return line.strip()
        # Fallback: look for a long, vivid sentence
        sentences = [s.strip() for s in reply.split(".") if len(s.strip().split()) > 7]
        if sentences:
            return sentences[0]
        return None
    if image_match:
        prompt = image_match.group(1).strip()
        if prompt:
            image_prompt = prompt
            plugin = PLUGINS.get("image_gen")
            if not plugin:
                return {"text": "Image generation plugin not loaded.", "image_url": "", "image_prompt": None}
            try:
                image_url = plugin.generate_image(prompt, user_id=req.user_id) or ""
            except Exception as e:
                return {"text": f"Error generating image: {e}", "image_url": "", "image_prompt": image_prompt}
            # Remove the [image: ...] tag and any prompt from the conversational reply
            text_reply = re.sub(r"\[image: .+?\]", "", text_reply, flags=re.IGNORECASE | re.DOTALL).strip()
        else:
            return {"text": "Sorry, I couldn't generate an image because the prompt was empty. Please try rephrasing your request.", "image_url": "", "image_prompt": None}
    elif has_image_intent(user_message):
        # Only trigger fallback if no [image: ...] tag
        creative_desc = extract_creative_description(llm_reply)
        if creative_desc:
            image_prompt = creative_desc
            plugin = PLUGINS.get("image_gen")
            if not plugin:
                return {"text": "Image generation plugin not loaded.", "image_url": "", "image_prompt": None}
            try:
                image_url = plugin.generate_image(image_prompt, user_id=req.user_id) or ""
            except Exception as e:
                return {"text": f"Error generating image: {e}", "image_url": "", "image_prompt": image_prompt}
        else:
            return {"text": "Sorry, I couldn't generate an image this time. Please try rephrasing your request.", "image_url": "", "image_prompt": None}
    # Store assistant response in context
    history.append({"role": "assistant", "content": llm_reply})
    USER_CONTEXT[req.user_id] = history
    final_text = ((text_reply or "") + (note_confirmation or "") + ("" if suppress_confirmation else todo_confirmation)).strip()
    if not final_text:
        final_text = "I'm not sure what you wanted to do. Please try again or rephrase your request."
    return {"text": final_text, "image_url": image_url, "image_prompt": image_prompt}

@app.post("/reset_context")
def reset_context(req: UserRequest):
    USER_CONTEXT[req.user_id] = []
    return {"status": "ok"}

@app.get("/get_context")
def get_context(user_id: str):
    history = USER_CONTEXT.get(user_id, [])
    model = USER_MODEL.get(user_id, DEFAULT_MODEL)
    # Token counting logic
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    # Compose message history as OpenAI expects
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    total_tokens = 0
    for msg in message_history:
        total_tokens += len(enc.encode(msg["content"]))
    # Model context window sizes (update as needed)
    model_context_windows = {
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385
    }
    context_window = model_context_windows.get(model, 8192)
    percent_used = (total_tokens / context_window) * 100
    return {
        "history": history,
        "message_count": len(history),
        "total_tokens": total_tokens,
        "context_window": context_window,
        "percent_used": percent_used,
        "model": model
    }

@app.post("/set_model")
def set_model(req: ModelRequest):
    if req.model not in SUPPORTED_MODELS:
        return {
            "error": f"Model '{req.model}' is not supported.",
            "available_models": SUPPORTED_MODELS,
            "usage": "Use /model <model_name> to change."
        }
    USER_MODEL[req.user_id] = req.model
    return {"status": "ok", "model": req.model}

@app.get("/get_model")
def get_model(user_id: str):
    model = USER_MODEL.get(user_id, DEFAULT_MODEL)
    return {"model": model}

@app.get("/models")
def get_models():
    return {"models": SUPPORTED_MODELS}

@app.get("/todo")
def get_todo(user_id: str):
    todo_plugin = PLUGINS.get("todo")
    if not todo_plugin:
        raise HTTPException(status_code=500, detail="To-do plugin not loaded.")
    result = todo_plugin.handle_todo("list", user_id)
    todos = todo_plugin.get_todo_list(user_id)
    return {"result": result, "todos": todos}

@app.post("/todo")
def add_todo_api(req: TodoRequest):
    todo_plugin = PLUGINS.get("todo")
    if not todo_plugin:
        raise HTTPException(status_code=500, detail="To-do plugin not loaded.")
    if not req.user_id or not isinstance(req.user_id, str):
        return JSONResponse(status_code=422, content={"detail": "user_id must be a string."})
    if not req.task or not isinstance(req.task, str) or not req.task.strip():
        return JSONResponse(status_code=422, content={"detail": "Missing or empty task."})
    result = todo_plugin.handle_todo("add", req.user_id, req.task)
    todos = todo_plugin.get_todo_list(req.user_id)
    return {"result": result, "todos": todos}

@app.post("/todo/done")
def mark_done_api(req: TodoRequest):
    todo_plugin = PLUGINS.get("todo")
    if not todo_plugin:
        raise HTTPException(status_code=500, detail="To-do plugin not loaded.")
    if not req.user_id or not isinstance(req.user_id, str):
        return JSONResponse(status_code=422, content={"detail": "user_id must be a string."})
    if req.query is None or not str(req.query).strip():
        return JSONResponse(status_code=422, content={"detail": "Missing or empty query (number or text)."})
    result = todo_plugin.handle_todo("done", req.user_id, req.query)
    todos = todo_plugin.get_todo_list(req.user_id)
    return {"result": result, "todos": todos}

@app.post("/todo/undone")
def unmark_done_api(req: TodoRequest):
    todo_plugin = PLUGINS.get("todo")
    if not todo_plugin:
        raise HTTPException(status_code=500, detail="To-do plugin not loaded.")
    if not req.user_id or not isinstance(req.user_id, str):
        return JSONResponse(status_code=422, content={"detail": "user_id must be a string."})
    if req.query is None or not str(req.query).strip():
        return JSONResponse(status_code=422, content={"detail": "Missing or empty query (number or text)."})
    result = todo_plugin.handle_todo("undone", req.user_id, req.query)
    todos = todo_plugin.get_todo_list(req.user_id)
    return {"result": result, "todos": todos}

@app.post("/todo/remove")
def remove_todo_api(req: TodoRequest):
    todo_plugin = PLUGINS.get("todo")
    if not todo_plugin:
        raise HTTPException(status_code=500, detail="To-do plugin not loaded.")
    if not req.user_id or not isinstance(req.user_id, str):
        return JSONResponse(status_code=422, content={"detail": "user_id must be a string."})
    if req.query is None or not str(req.query).strip():
        return JSONResponse(status_code=422, content={"detail": "Missing or empty query (number or text)."})
    result = todo_plugin.handle_todo("remove", req.user_id, req.query)
    todos = todo_plugin.get_todo_list(req.user_id)
    return {"result": result, "todos": todos} 