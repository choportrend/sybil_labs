import requests

GOOGLE_API_KEY = "AIzaSyDpbyMLlaO3sGHfUd9GGpK9Whi0R2-xGN4"
GOOGLE_CX = "074d541947d2b4111"

class WebSearchPlugin:
    name = "web_search"

    def search_web(self, query, user_id=None):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX,
            "q": query,
            "num": 3,
        }
        try:
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            if "items" in data:
                results = data["items"]
                summary = []
                for item in results[:3]:
                    title = item.get("title", "")
                    snippet = item.get("snippet", "")
                    link = item.get("link", "")
                    summary.append(f"- {title}: {snippet} ({link})")
                return "\n".join(summary)
            return "No relevant web result found."
        except Exception as e:
            return f"Web search error: {e}"

def register_plugin():
    return WebSearchPlugin() 