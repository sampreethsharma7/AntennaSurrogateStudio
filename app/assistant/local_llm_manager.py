class LocalLLMManager:
    MODES = ["Basic Offline Guide", "Lite Local LLM", "Standard Local LLM", "Advanced Local LLM"]

    def available_models(self) -> list:
        return []

    def status(self) -> str:
        return "No local LLM provider is configured. Basic Offline Guide remains available."
