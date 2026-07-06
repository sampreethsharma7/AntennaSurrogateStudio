from pathlib import Path

from app.assistant.base_backend import AssistantBackend
from app.assistant.documentation_search import DocumentationSearch


class OfflineGuideBackend(AssistantBackend):
    def __init__(self, knowledge_base_dir: Path):
        self.searcher = DocumentationSearch(knowledge_base_dir)

    def is_available(self) -> bool:
        return True

    def get_status(self) -> str:
        return "Basic Offline Guide is available. No cloud service or API key is used."

    def answer(self, question: str, app_context: dict) -> str:
        q = question.lower()
        if "why" in q and "train" in q and not app_context.get("workflow_completion", {}).get("prepared"):
            return "The Train Model step is locked until a prepared dataset is saved. Go to Import & Configure Data, select at least one input and one output column, then click Prepare Dataset."
        if "rmse" in q:
            return "RMSE is the root mean squared error. It summarizes the typical prediction error magnitude on the held-out test set. Lower values mean predictions are closer to the actual outputs."
        if "r2" in q or "r²" in q:
            return "R² measures how much variation in the held-out outputs is explained by the model. Values closer to 1 are usually better, but it should be read alongside RMSE, plots, and data quality warnings."
        if "extrapolation" in q or "outside" in q:
            return "Extrapolation means a prediction uses input values outside the range observed during training. The app allows the prediction, but flags it so you treat the result cautiously."
        hits = self.searcher.search(question)
        if hits:
            return "\n\n".join([f"{title}\n{snippet}" for title, snippet in hits])
        return "I could not find that feature in the current version of Antenna Surrogate Studio. Check the Help page or contact the developer for confirmation."

    def topics(self) -> list[str]:
        return [title for title, _ in self.searcher.documents()]
