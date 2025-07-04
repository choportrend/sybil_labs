import os
import json
import difflib
from .todo_data import add_todo, list_todos, mark_done, unmark_done, remove_todo

class TodoPlugin:
    name = "todo"

    def _find_task(self, user_id, query):
        todos = list_todos(user_id)
        # Try number
        try:
            idx = int(query) - 1
            if 0 <= idx < len(todos):
                return idx, todos[idx], None
        except Exception:
            pass
        # Fuzzy match by text
        query = query.lower().strip()
        matches = [(i, item) for i, item in enumerate(todos) if query in item["task"].lower()]
        if len(matches) == 1:
            return matches[0][0], matches[0][1], None
        elif len(matches) > 1:
            return None, None, f"Ambiguous task description. Multiple tasks match '{query}'. Please specify the number."
        # Use difflib for best match if nothing else
        best = None
        best_score = 0.0
        for i, item in enumerate(todos):
            score = difflib.SequenceMatcher(None, query, item["task"].lower()).ratio()
            if score > best_score:
                best_score = score
                best = (i, item)
        if best and best_score > 0.7:
            return best[0], best[1], None
        return None, None, f"Task '{query}' not found. Please specify the number or a more exact description."

    def handle_todo(self, action, user_id, args=None):
        # Always reload the list after each action
        def reload():
            return list_todos(user_id)
        todos = reload()
        if action == "add":
            add_todo(user_id, args)
            return f"I've added '{args}' to your to-do list! Let me know if you need anything else."
        elif action == "list":
            if not todos:
                return "Your to-do list is currently empty. Ready to get things done?"
            msg = "Here's your to-do list:\n"
            for i, item in enumerate(todos):
                status = "✅" if item["done"] else "❌"
                msg += f"{i+1}. {status} {item['task']}\n"
            return msg.strip()
        elif action == "done":
            idx, task, err = self._find_task(user_id, args)
            if err:
                return err
            todos = reload()
            if idx is not None and mark_done(user_id, idx):
                return f"Great job! I've marked task {idx+1}: '{task['task']}' as done."
            else:
                return "Hmm, I couldn't find that task to mark as done. Could you try rephrasing or give me the number?"
        elif action == "undone":
            idx, task, err = self._find_task(user_id, args)
            if err:
                return err
            todos = reload()
            if idx is not None and unmark_done(user_id, idx):
                return f"I've unmarked task {idx+1}: '{task['task']}' as done. Keep going, you got this!"
            else:
                return "I couldn't find that task to unmark. Could you try rephrasing or give me the number?"
        elif action == "remove":
            idx, task, err = self._find_task(user_id, args)
            if err:
                return err
            todos = reload()
            if idx is not None and remove_todo(user_id, idx):
                return f"I've removed task {idx+1}: '{task['task']}' from your to-do list."
            else:
                return "I couldn't find that task to remove. Could you try rephrasing or give me the number?"
        else:
            return "I'm not sure what you want to do with your to-do list. Could you clarify?"

    def get_todo_list(self, user_id):
        return list_todos(user_id)

def register_plugin():
    return TodoPlugin() 