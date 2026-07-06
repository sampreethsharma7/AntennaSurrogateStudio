from abc import ABC, abstractmethod


class AssistantBackend(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_status(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def answer(self, question: str, app_context: dict) -> str:
        raise NotImplementedError

    def list_models(self) -> list:
        return []

    def select_model(self, model_id: str) -> None:
        raise ValueError("Local LLM model selection is not available in this backend.")
