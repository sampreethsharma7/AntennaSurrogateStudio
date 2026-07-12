import json
import urllib.error
import urllib.request

from app.assistant.base_backend import AssistantBackend

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:1.7b"
STATUS_TIMEOUT_SECONDS = 3
GENERATE_TIMEOUT_SECONDS = 180

SYSTEM_PROMPT = (
    "You are the offline help assistant built into Antenna Surrogate Studio, a desktop app for antenna/RF "
    "engineers to train XGBoost surrogate models from CSV simulation data. Answer only using the product "
    "documentation below. If the documentation does not cover the question, say so plainly instead of "
    "guessing. Do not give antenna design or RF engineering advice, only product-usage help. Keep answers "
    "concise. Respond in English only.\n\n--- PRODUCT DOCUMENTATION ---\n{context}\n--- END DOCUMENTATION ---"
)


class LocalLLMBackend(AssistantBackend):
    def __init__(self, context_text: str, model_name: str = MODEL_NAME):
        self.context_text = context_text
        self.model_name = model_name
        self._availability_cache = None

    def is_available(self) -> bool:
        if self._availability_cache is not None:
            return self._availability_cache
        try:
            request = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
            with urllib.request.urlopen(request, timeout=STATUS_TIMEOUT_SECONDS) as response:
                data = json.loads(response.read())
            names = {model.get("name") for model in data.get("models", [])}
            self._availability_cache = self.model_name in names
        except (urllib.error.URLError, OSError, ValueError):
            self._availability_cache = False
        return self._availability_cache

    def get_status(self) -> str:
        if self.is_available():
            return f"Local LLM ({self.model_name}) is running via Ollama."
        return f"Local LLM is not available. Install Ollama and run 'ollama pull {self.model_name}', or use the Basic Offline Guide."

    def answer(self, question: str, app_context: dict) -> str:
        system = SYSTEM_PROMPT.format(context=self.context_text)
        prompt = f"{system}\n\n--- QUESTION ---\n/no_think\n{question}"
        payload = json.dumps({"model": self.model_name, "prompt": prompt, "stream": False}).encode("utf-8")
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate", data=payload, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=GENERATE_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read())
        return data.get("response", "").strip() or "The local model returned an empty response."
