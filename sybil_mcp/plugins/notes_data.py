import os
import json
from datetime import datetime

NOTES_FILE = os.path.join(os.path.dirname(__file__), '../config/notes_data.json')

def load_notes():
    if not os.path.exists(NOTES_FILE):
        return {}
    with open(NOTES_FILE, 'r') as f:
        return json.load(f)

def save_notes(notes):
    with open(NOTES_FILE, 'w') as f:
        json.dump(notes, f)

def get_user_notes(user_id):
    notes = load_notes()
    return notes.get(str(user_id), [])

def add_note(user_id, text):
    notes = load_notes()
    user_notes = notes.get(str(user_id), [])
    note = {
        "text": text,
        "created": datetime.now().isoformat()
    }
    user_notes.append(note)
    notes[str(user_id)] = user_notes
    save_notes(notes)

def list_notes(user_id):
    return get_user_notes(user_id)

def remove_note(user_id, index):
    notes = load_notes()
    user_notes = notes.get(str(user_id), [])
    if 0 <= index < len(user_notes):
        user_notes.pop(index)
        notes[str(user_id)] = user_notes
        save_notes(notes)
        return True
    return False

def search_notes(user_id, query):
    user_notes = get_user_notes(user_id)
    query = query.lower().strip()
    return [n for n in user_notes if query in n["text"].lower()]

def show_note(user_id, index):
    user_notes = get_user_notes(user_id)
    if 0 <= index < len(user_notes):
        return user_notes[index]
    return None 