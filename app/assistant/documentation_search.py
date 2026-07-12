from pathlib import Path


class DocumentationSearch:
    def __init__(self, knowledge_base_dir: Path):
        self.knowledge_base_dir = knowledge_base_dir

    def documents(self) -> list[tuple[str, str]]:
        docs = []
        for path in sorted(self.knowledge_base_dir.glob("*.md")):
            docs.append((path.stem.replace("_", " ").title(), path.read_text(encoding="utf-8")))
        return docs

    def search(self, query: str, limit: int = 5) -> list[tuple[str, str]]:
        terms = [term.lower() for term in query.split() if len(term) > 2]
        scored = []
        for title, text in self.documents():
            lower = text.lower()
            score = sum(lower.count(term) for term in terms)
            if score:
                snippet = text.strip().split("\n\n")[0][:500]
                scored.append((score, title, snippet))
        scored.sort(reverse=True)
        return [(title, snippet) for _, title, snippet in scored[:limit]]

    def combined_text(self, manual_path: Path = None) -> str:
        parts = []
        if manual_path and manual_path.exists():
            parts.append(manual_path.read_text(encoding="utf-8"))
        for title, text in self.documents():
            parts.append(f"## {title}\n{text}")
        return "\n\n".join(parts)
