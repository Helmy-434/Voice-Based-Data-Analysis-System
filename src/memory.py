class Memory:
    def __init__(self):
        self.data = {
            "preferences": {},
            "turns": [],
            "last_code": ""
        }

    def render(self) -> str:
        parts = []
        if self.data.get("preferences"):
            prefs_str = ", ".join(f"{k}: {v}" for k, v in self.data["preferences"].items())
            parts.append(f"User preferences: {prefs_str}")
        if self.data.get("last_code"):
            parts.append(f"Last generated code:\n```python\n{self.data['last_code']}\n```")
        if self.data.get("turns"):
            recent_turns = self.data["turns"][-4:]
            turns_repr = []
            for t in recent_turns:
                user_msg = t.get("user", "")
                assistant_msg = t.get("assistant", "")[:300]
                turns_repr.append(f"User: {user_msg}\nAssistant: {assistant_msg}")
            parts.append("Recent conversation turns:\n" + "\n---\n".join(turns_repr))
        return "\n\n".join(parts) if parts else "No prior context."

    def update_preferences(self, prompt: str):
        p_lower = prompt.lower()
        if "python 3" in p_lower:
            self.data["preferences"]["python_version"] = "Python 3"
        if "type hints" in p_lower or "typed" in p_lower:
            self.data["preferences"]["type_hints"] = True
        if "docstring" in p_lower:
            self.data["preferences"]["docstrings"] = True
        if "fastapi" in p_lower:
            self.data["preferences"]["framework"] = "FastAPI"
        elif "flask" in p_lower:
            self.data["preferences"]["framework"] = "Flask"
        elif "django" in p_lower:
            self.data["preferences"]["framework"] = "Django"
        elif "pytorch" in p_lower:
            self.data["preferences"]["framework"] = "PyTorch"
        if "tabs" in p_lower:
            self.data["preferences"]["indentation"] = "tabs"
        elif "spaces" in p_lower:
            self.data["preferences"]["indentation"] = "spaces"

    def add_turn(self, user: str, assistant: str, code: str = ""):
        turn = {"user": user, "assistant": assistant, "code": code}
        self.data["turns"].append(turn)
        if code:
            self.data["last_code"] = code

    def clear(self):
        self.data = {
            "preferences": {},
            "turns": [],
            "last_code": ""
        }