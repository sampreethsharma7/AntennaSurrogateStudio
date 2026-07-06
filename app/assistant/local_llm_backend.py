from app.assistant.base_backend import AssistantBackend


class LocalLLMBackend(AssistantBackend):
    def is_available(self) -> bool:
        return False

    def get_status(self) -> str:
        return "Local LLM backend abstraction is present, but no provider is configured in v1."

    def answer(self, question: str, app_context: dict) -> str:
        return "Local LLM mode is not configured. Use Basic Offline Guide for product help."
