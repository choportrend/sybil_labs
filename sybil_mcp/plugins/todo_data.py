import os
import json

TODO_FILE = os.path.join(os.path.dirname(__file__), '../config/todo_data.json')


def load_todos():
    if not os.path.exists(TODO_FILE):
        return {}
    with open(TODO_FILE, 'r') as f:
        return json.load(f)

def save_todos(todos):
    with open(TODO_FILE, 'w') as f:
        json.dump(todos, f)

def get_user_todos(user_id):
    todos = load_todos()
    return todos.get(str(user_id), [])

def add_todo(user_id, task):
    todos = load_todos()
    user_todos = todos.get(str(user_id), [])
    # Deduplicate: do not add if a task with the same name exists (case-insensitive, trimmed)
    task_clean = task.strip().lower()
    for t in user_todos:
        if t["task"].strip().lower() == task_clean:
            return  # Do not add duplicate
    user_todos.append({"task": task, "done": False})
    todos[str(user_id)] = user_todos
    save_todos(todos)

def list_todos(user_id):
    return get_user_todos(user_id)

def mark_done(user_id, index):
    todos = load_todos()
    user_todos = todos.get(str(user_id), [])
    if 0 <= index < len(user_todos):
        user_todos[index]["done"] = True
        todos[str(user_id)] = user_todos
        save_todos(todos)
        return True
    return False

def unmark_done(user_id, index):
    todos = load_todos()
    user_todos = todos.get(str(user_id), [])
    if 0 <= index < len(user_todos):
        user_todos[index]["done"] = False
        todos[str(user_id)] = user_todos
        save_todos(todos)
        return True
    return False

def remove_todo(user_id, index):
    todos = load_todos()
    user_todos = todos.get(str(user_id), [])
    if 0 <= index < len(user_todos):
        removed = user_todos.pop(index)
        todos[str(user_id)] = user_todos
        save_todos(todos)
        return True
    return False 