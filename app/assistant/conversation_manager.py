from pathlib import Path
import json


class ConversationManager:
    def __init__(self, history_path: Path):
        self.history_path = history_path

    def load(self) -> list[dict]:
        if not self.history_path.exists():
            return []
        return json.loads(self.history_path.read_text(encoding="utf-8"))

    def append(self, role: str, content: str) -> None:
        history = self.load()
        history.append({"role": role, "content": content})
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.history_path.write_text("[]", encoding="utf-8")
