import unittest

from studio.planner_model_discovery import (
    GEMINI,
    GROQ,
    LOCAL_OLLAMA,
    OPENROUTER,
    parse_gemini_models,
    parse_groq_models,
    parse_ollama_models,
    parse_openrouter_models,
)


class PlannerModelDiscoveryTests(unittest.TestCase):
    def test_ollama_filters_embedding_models_and_marks_validated_qwen(self):
        models = parse_ollama_models({"models": [
            {"name": "qwen3:8b", "details": {"family": "qwen3"}},
            {"name": "nomic-embed-text:latest", "details": {"family": "bert"}},
            {"name": "llama3.2:latest", "details": {"family": "llama"}},
        ]})
        self.assertEqual({item.model_id for item in models}, {"qwen3:8b", "llama3.2:latest"})
        self.assertEqual(models[0].provider, LOCAL_OLLAMA)
        self.assertTrue(next(item for item in models if item.model_id == "qwen3:8b").tested)
        self.assertIn("Untested", next(item for item in models if item.model_id.startswith("llama")).menu_label)

    def test_gemini_requires_generate_content(self):
        models = parse_gemini_models({"models": [
            {"name": "models/gemini-3.8-flash", "baseModelId": "gemini-3.8-flash", "displayName": "Gemini 3.8 Flash", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/text-embedding-004", "supportedGenerationMethods": ["embedContent"]},
        ]})
        self.assertEqual([item.model_id for item in models], ["gemini-3.8-flash"])
        self.assertEqual(models[0].provider, GEMINI)
        self.assertTrue(models[0].tested)

    def test_groq_filters_audio_and_inactive_models(self):
        models = parse_groq_models({"data": [
            {"id": "openai/gpt-oss-120b", "active": True},
            {"id": "whisper-large-v3", "active": True},
            {"id": "retired-chat", "active": False},
            {"id": "llama-3.3-70b-versatile", "active": True},
        ]})
        self.assertEqual({item.model_id for item in models}, {"openai/gpt-oss-120b", "llama-3.3-70b-versatile"})
        self.assertEqual(models[0].provider, GROQ)
        self.assertTrue(next(item for item in models if item.model_id == "openai/gpt-oss-120b").tested)

    def test_openrouter_uses_modality_and_pricing_for_free_filter(self):
        payload = {"data": [
            {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "name": "Nemotron", "architecture": {"modality": "text->text", "output_modalities": ["text"]}, "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "vendor/paid-chat", "architecture": {"modality": "text->text", "output_modalities": ["text"]}, "pricing": {"prompt": "0.1", "completion": "0.2"}},
            {"id": "vendor/image-only", "architecture": {"modality": "text->image", "output_modalities": ["image"]}, "pricing": {"prompt": "0", "completion": "0"}},
        ]}
        free = parse_openrouter_models(payload, free_only=True)
        all_models = parse_openrouter_models(payload, free_only=False)
        self.assertEqual([item.model_id for item in free], ["nvidia/nemotron-3-ultra-550b-a55b:free"])
        self.assertEqual({item.model_id for item in all_models}, {"nvidia/nemotron-3-ultra-550b-a55b:free", "vendor/paid-chat"})
        self.assertEqual(free[0].provider, OPENROUTER)
        self.assertTrue(free[0].tested)


if __name__ == "__main__":
    unittest.main()
