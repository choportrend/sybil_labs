import difflib
from .notes_data import add_note, list_notes, remove_note, search_notes, show_note

class NotesPlugin:
    name = "notes"

    def _find_note(self, user_id, query):
        notes = list_notes(user_id)
        # Try number
        try:
            idx = int(query) - 1
            if 0 <= idx < len(notes):
                return idx, notes[idx], None
        except Exception:
            pass
        # Fuzzy match by text
        query = query.lower().strip()
        matches = [(i, item) for i, item in enumerate(notes) if query in item["text"].lower()]
        if len(matches) == 1:
            return matches[0][0], matches[0][1], None
        elif len(matches) > 1:
            return None, None, f"Ambiguous note description. Multiple notes match '{query}'. Please specify the number."
        # Use difflib for best match if nothing else
        best = None
        best_score = 0.0
        for i, item in enumerate(notes):
            score = difflib.SequenceMatcher(None, query, item["text"].lower()).ratio()
            if score > best_score:
                best_score = score
                best = (i, item)
        if best and best_score > 0.7:
            return best[0], best[1], None
        return None, None, f"Note '{query}' not found. Please specify the number or a more exact description."

    def handle_note(self, action, user_id, args=None):
        def reload():
            return list_notes(user_id)
        notes = reload()
        if action == "add":
            add_note(user_id, args)
            return f"Noted: '{args}'"
        elif action == "list":
            if not notes:
                return "You have no notes yet. Ready to jot something down?"
            msg = "Here are your notes:\n"
            for i, item in enumerate(notes):
                msg += f"{i+1}. {item['text']} (added {item['created'][:16].replace('T',' ')})\n"
            return msg.strip()
        elif action == "search":
            found = search_notes(user_id, args)
            if not found:
                return f"No notes found matching '{args}'."
            msg = "Found these notes:\n"
            for i, item in enumerate(found):
                msg += f"- {item['text']} (added {item['created'][:16].replace('T',' ')})\n"
            return msg.strip()
        elif action == "show":
            idx, note, err = self._find_note(user_id, args)
            if err:
                return err
            if idx is not None:
                return f"Note {idx+1}: {note['text']} (added {note['created'][:16].replace('T',' ')})"
            else:
                return "I couldn't find that note. Could you try rephrasing or give me the number?"
        elif action == "remove":
            idx, note, err = self._find_note(user_id, args)
            if err:
                return err
            if idx is not None and remove_note(user_id, idx):
                return f"I've removed note {idx+1}: '{note['text']}'"
            else:
                return "I couldn't find that note to remove. Could you try rephrasing or give me the number?"
        else:
            return "I'm not sure what you want to do with your notes. Could you clarify?"

    def get_notes(self, user_id):
        return list_notes(user_id)

def register_plugin():
    return NotesPlugin() 