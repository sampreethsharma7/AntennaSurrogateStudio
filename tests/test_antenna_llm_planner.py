import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from studio.antenna_agent import create_default_agent
from studio.antenna_llm_planner import (
    LLMToolPlan,
    GeminiSchemaConstrainedPlanner,
    GroqSchemaConstrainedPlanner,
    OpenRouterNemotronPlanner,
    PlannedToolCall,
    SchemaConstrainedLLMPlanner,
    build_planner_exchange,
    enforce_request_capabilities,
    load_gemini_api_key,
    load_groq_api_key,
    load_openrouter_api_key,
    parse_llm_tool_plan,
)
from studio.antenna_tools import CapabilityError


class _Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.payload


class LLMPlannerTests(unittest.TestCase):
    def setUp(self):
        self.agent = create_default_agent()
        self.state = self.agent.create_design("inset_patch")
        self.manifest = self.agent.capability_manifest(self.state)
        self.names = tuple(item["name"] for item in self.manifest["callable_tools"])

    def test_strict_schema_accepts_registered_ordered_calls(self):
        parsed = parse_llm_tool_plan({
            "schema_version": 1,
            "status": "execute",
            "message": "Create and parameterize a circular patch.",
            "calls": [
                {"name": "recipe.select", "arguments": {"recipe_id": "circular_patch_v1"}},
                {"name": "parameter.set", "arguments": {"key": "frequency_ghz", "value": 5.8}},
            ],
        }, callable_tool_names=self.names)
        self.assertEqual([call.name for call in parsed.calls], ["recipe.select", "parameter.set"])

    def test_schema_rejects_unregistered_calls_extra_fields_and_calls_on_refusal(self):
        with self.assertRaisesRegex(CapabilityError, "unregistered"):
            parse_llm_tool_plan({"schema_version": 1, "status": "execute", "message": "", "calls": [{"name": "Solver.Start", "arguments": {}}]}, callable_tool_names=self.names)
        with self.assertRaisesRegex(CapabilityError, "schema"):
            parse_llm_tool_plan({"schema_version": 1, "status": "execute", "message": "", "calls": [], "code": "x"}, callable_tool_names=self.names)
        with self.assertRaisesRegex(CapabilityError, "cannot contain"):
            parse_llm_tool_plan({"schema_version": 1, "status": "refuse", "message": "unsupported", "calls": [{"name": "design.reset", "arguments": {}}]}, callable_tool_names=self.names)
        with self.assertRaisesRegex(CapabilityError, "more than once"):
            parse_llm_tool_plan({
                "schema_version": 1,
                "status": "execute",
                "message": "duplicate",
                "calls": [
                    {"name": "parameter.set", "arguments": {"key": "array_columns", "value": 2}},
                    {"name": "parameter.set", "arguments": {"key": "array_columns", "value": 4}},
                ],
            }, callable_tool_names=self.names)

    def test_local_planner_sends_current_state_runtime_tools_and_json_schema(self):
        captured = {}

        def opener(request, timeout):
            captured["timeout"] = timeout
            captured["body"] = json.loads(request.data.decode("utf-8"))
            plan = {"schema_version": 1, "status": "execute", "message": "resize", "calls": [{"name": "parameter.set", "arguments": {"key": "patch_width_mm", "value": 40}}]}
            return _Response({"message": {"content": json.dumps(plan)}})

        planner = SchemaConstrainedLLMPlanner("test-model", base_url="http://127.0.0.1:11434", opener=opener)
        result = planner.plan(instruction="Make the patch wider", current_design=self.state.to_dict(), capability_manifest=self.manifest)
        self.assertEqual(result.calls[0].name, "parameter.set")
        body = captured["body"]
        call_variants = body["format"]["properties"]["calls"]["items"]["oneOf"]
        self.assertEqual(
            [variant["properties"]["name"]["const"] for variant in call_variants],
            list(self.names),
        )
        parameter_call = next(
            variant for variant in call_variants
            if variant["properties"]["name"]["const"] == "parameter.set"
        )
        self.assertEqual(
            parameter_call["properties"]["arguments"]["required"],
            ["key", "value"],
        )
        self.assertFalse(parameter_call["properties"]["arguments"]["additionalProperties"])
        user_payload = json.loads(body["messages"][1]["content"])
        self.assertEqual(user_payload["current_design"]["design_id"], self.state.design_id)
        self.assertIn("deterministic_primitive_tools", user_payload["runtime_capabilities"])
        self.assertFalse(body["stream"])
        self.assertEqual(body["options"]["temperature"], 0.0)
        self.assertEqual(body["options"]["num_ctx"], 16384)

    def test_invalid_first_plan_gets_one_schema_repair_attempt(self):
        requests = []
        responses = iter((
            {
                "schema_version": 1,
                "status": "refuse",
                "message": "",
                "calls": [{"name": "design.reset", "arguments": {}}],
            },
            {
                "schema_version": 1,
                "status": "execute",
                "message": "Reset the current design.",
                "calls": [{"name": "design.reset", "arguments": {}}],
            },
        ))

        def opener(request, _timeout=None, **_kwargs):
            requests.append(json.loads(request.data.decode("utf-8")))
            return _Response({"message": {"content": json.dumps(next(responses))}})

        planner = SchemaConstrainedLLMPlanner(
            "test-model",
            base_url="http://127.0.0.1:11434",
            opener=opener,
        )
        result = planner.plan(
            instruction="Reset this design",
            current_design=self.state.to_dict(),
            capability_manifest=self.manifest,
        )

        self.assertEqual(result.status, "execute")
        self.assertEqual(len(requests), 2)
        self.assertIn("failed deterministic validation", requests[1]["messages"][-1]["content"])

    def test_gemini_uses_same_prompt_context_and_post_generation_contract(self):
        local_request = {}
        gemini_request = {}
        plan = {
            "schema_version": 1,
            "status": "execute",
            "message": "resize",
            "calls": [{"name": "parameter.set", "arguments": {"key": "patch_width_mm", "value": 40}}],
        }

        def local_opener(request, timeout):
            local_request.update(
                timeout=timeout,
                body=json.loads(request.data.decode("utf-8")),
            )
            return _Response({"message": {"content": json.dumps(plan)}})

        def gemini_opener(request, timeout):
            gemini_request.update(
                timeout=timeout,
                url=request.full_url,
                key=request.get_header("X-goog-api-key"),
                body=json.loads(request.data.decode("utf-8")),
            )
            return _Response({
                "candidates": [{"content": {"parts": [{"text": json.dumps(plan)}]}}]
            })

        local = SchemaConstrainedLLMPlanner(
            "test-local", base_url="http://127.0.0.1:11434", opener=local_opener
        )
        gemini = GeminiSchemaConstrainedPlanner(
            "test-gemini", api_key="secret-for-test", opener=gemini_opener
        )
        request = {
            "instruction": "Make the patch wider",
            "current_design": self.state.to_dict(),
            "capability_manifest": self.manifest,
        }

        self.assertEqual(local.plan(**request), gemini.plan(**request))
        local_body = local_request["body"]
        cloud_body = gemini_request["body"]
        self.assertEqual(
            local_body["messages"][0]["content"],
            cloud_body["systemInstruction"]["parts"][0]["text"],
        )
        self.assertEqual(
            local_body["messages"][1]["content"],
            cloud_body["contents"][0]["parts"][0]["text"],
        )
        shared_payload = json.loads(local_body["messages"][1]["content"])
        self.assertEqual(local_body["format"], shared_payload["tool_plan_schema"])
        self.assertNotIn("responseJsonSchema", cloud_body["generationConfig"])
        self.assertEqual(cloud_body["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(
            gemini.last_run_metadata["schema_enforcement"],
            "post_generation_strict_parsing_and_validation",
        )
        self.assertEqual(gemini_request["key"], "secret-for-test")
        self.assertNotIn("secret-for-test", gemini_request["url"])
        self.assertNotIn("secret-for-test", json.dumps(cloud_body))

    def test_gemini_missing_key_is_clear_and_local_planner_remains_independent(self):
        with patch.dict(os.environ, {}, clear=True):
            planner = GeminiSchemaConstrainedPlanner(api_key="", env_file=Path("missing.env"))
            with self.assertRaisesRegex(CapabilityError, "GEMINI_API_KEY.*Local Qwen remains available"):
                planner.plan(
                    instruction="Make the patch wider",
                    current_design=self.state.to_dict(),
                    capability_manifest=self.manifest,
                )

    def test_gemini_key_can_be_loaded_from_ignored_env_file(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("# local only\nGEMINI_API_KEY='file-key'\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_gemini_api_key(env_file=env_file), "file-key")

    def test_gemini_post_generation_validation_gets_one_schema_repair(self):
        requests = []
        responses = iter((
            {"schema_version": 1, "status": "refuse", "message": "bad", "calls": [
                {"name": "design.reset", "arguments": {}}
            ]},
            {"schema_version": 1, "status": "execute", "message": "reset", "calls": [
                {"name": "design.reset", "arguments": {}}
            ]},
        ))

        def opener(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            plan = next(responses)
            return _Response({
                "candidates": [{"content": {"parts": [{"text": json.dumps(plan)}]}}]
            })

        planner = GeminiSchemaConstrainedPlanner(
            "test-gemini", api_key="secret-for-test", opener=opener
        )
        result = planner.plan(
            instruction="Reset this design",
            current_design=self.state.to_dict(),
            capability_manifest=self.manifest,
        )

        self.assertEqual(result.status, "execute")
        self.assertEqual(len(requests), 2)
        self.assertNotIn("responseJsonSchema", requests[0]["generationConfig"])
        self.assertIn(
            "failed deterministic validation",
            requests[1]["contents"][-1]["parts"][0]["text"],
        )

    def test_groq_uses_same_exchange_and_unchanged_strict_schema(self):
        captured = {}
        plan = {
            "schema_version": 1,
            "status": "execute",
            "message": "resize",
            "calls": [{"name": "parameter.set", "arguments": {"key": "patch_width_mm", "value": 40}}],
        }

        def opener(request, timeout):
            captured.update(
                timeout=timeout,
                url=request.full_url,
                authorization=request.get_header("Authorization"),
                body=json.loads(request.data.decode("utf-8")),
            )
            return _Response({"choices": [{"message": {"content": json.dumps(plan)}}]})

        planner = GroqSchemaConstrainedPlanner(
            api_key="secret-for-test", opener=opener
        )
        result = planner.plan(
            instruction="Make the patch wider",
            current_design=self.state.to_dict(),
            capability_manifest=self.manifest,
        )

        self.assertEqual(result.calls[0].name, "parameter.set")
        exchange = build_planner_exchange(
            instruction="Make the patch wider",
            current_design=self.state.to_dict(),
            capability_manifest=self.manifest,
        )
        body = captured["body"]
        self.assertEqual(body["messages"], [
            {"role": "system", "content": exchange.system_instruction},
            {"role": "user", "content": exchange.user_content},
        ])
        self.assertEqual(body["response_format"], {
            "type": "json_schema",
            "json_schema": {
                "name": "antenna_tool_plan",
                "strict": True,
                "schema": exchange.schema,
            },
        })
        self.assertEqual(body["model"], "openai/gpt-oss-120b")
        self.assertEqual(body["temperature"], 0.0)
        self.assertFalse(body["stream"])
        self.assertEqual(body["max_completion_tokens"], 1024)
        self.assertEqual(captured["authorization"], "Bearer secret-for-test")
        self.assertNotIn("secret-for-test", captured["url"])
        self.assertNotIn("secret-for-test", json.dumps(body))
        self.assertEqual(
            planner.last_run_metadata["schema_enforcement"],
            "provider_strict_json_schema_and_post_generation_validation",
        )
        self.assertFalse(planner.last_run_metadata["strict_schema_fallback"])

    def test_groq_exact_schema_rejection_falls_back_without_weakening_schema(self):
        requests = []
        plan = {
            "schema_version": 1,
            "status": "refuse",
            "message": "The requested edit is unavailable.",
            "calls": [],
        }

        def opener(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            if len(requests) == 1:
                error = {"error": {"message": "response_format json_schema contains an unsupported schema keyword"}}
                raise urllib.error.HTTPError(
                    request.full_url,
                    400,
                    "Bad Request",
                    {},
                    io.BytesIO(json.dumps(error).encode("utf-8")),
                )
            return _Response({"choices": [{"message": {"content": json.dumps(plan)}}]})

        planner = GroqSchemaConstrainedPlanner(
            api_key="secret-for-test", opener=opener
        )
        result = planner.plan(
            instruction="Remove the circular slot",
            current_design=self.state.to_dict(),
            capability_manifest=self.manifest,
        )

        exchange = build_planner_exchange(
            instruction="Remove the circular slot",
            current_design=self.state.to_dict(),
            capability_manifest=self.manifest,
        )
        self.assertEqual(result.status, "refuse")
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            requests[0]["response_format"]["json_schema"]["schema"],
            exchange.schema,
        )
        self.assertTrue(requests[0]["response_format"]["json_schema"]["strict"])
        self.assertEqual(requests[1]["response_format"], {"type": "json_object"})
        self.assertEqual(requests[0]["messages"], requests[1]["messages"])
        self.assertEqual(
            planner.last_run_metadata["schema_enforcement"],
            "json_object_post_generation_strict_parsing_and_validation",
        )
        self.assertTrue(planner.last_run_metadata["strict_schema_fallback"])

    def test_groq_key_can_be_loaded_and_missing_key_is_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("GROQ_API_KEY='file-key'\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_groq_api_key(env_file=env_file), "file-key")
                planner = GroqSchemaConstrainedPlanner(api_key="", env_file=Path(directory) / "missing.env")
                with self.assertRaisesRegex(CapabilityError, "GROQ_API_KEY.*Local Qwen remains available"):
                    planner.plan(
                        instruction="Make the patch wider",
                        current_design=self.state.to_dict(),
                        capability_manifest=self.manifest,
                    )

    def test_openrouter_nemotron_uses_same_exchange_and_local_strict_validation(self):
        captured = {}
        plan = {
            "schema_version": 1,
            "status": "execute",
            "message": "resize",
            "calls": [{"name": "parameter.set", "arguments": {"key": "patch_width_mm", "value": 40}}],
        }

        def opener(request, timeout):
            captured.update(
                timeout=timeout,
                url=request.full_url,
                authorization=request.get_header("Authorization"),
                body=json.loads(request.data.decode("utf-8")),
            )
            return _Response({"choices": [{"message": {"content": json.dumps(plan)}}]})

        planner = OpenRouterNemotronPlanner(api_key="secret-for-test", opener=opener)
        result = planner.plan(
            instruction="Make the patch wider",
            current_design=self.state.to_dict(),
            capability_manifest=self.manifest,
        )

        exchange = build_planner_exchange(
            instruction="Make the patch wider",
            current_design=self.state.to_dict(),
            capability_manifest=self.manifest,
        )
        self.assertEqual(result.calls[0].name, "parameter.set")
        self.assertEqual(captured["body"]["messages"], [
            {"role": "system", "content": exchange.system_instruction},
            {"role": "user", "content": exchange.user_content},
        ])
        self.assertEqual(
            json.loads(captured["body"]["messages"][1]["content"])["tool_plan_schema"],
            exchange.schema,
        )
        self.assertEqual(
            captured["body"]["model"],
            "nvidia/nemotron-3-ultra-550b-a55b:free",
        )
        self.assertNotIn("response_format", captured["body"])
        self.assertEqual(captured["authorization"], "Bearer secret-for-test")
        self.assertNotIn("secret-for-test", captured["url"])
        self.assertNotIn("secret-for-test", json.dumps(captured["body"]))
        self.assertEqual(
            planner.last_run_metadata["schema_enforcement"],
            "post_generation_strict_parsing_and_validation_no_provider_schema_enforcement",
        )

    def test_openrouter_nemotron_invalid_output_gets_one_schema_repair(self):
        requests = []
        responses = iter((
            {"schema_version": 1, "status": "refuse", "message": "bad", "calls": [
                {"name": "design.reset", "arguments": {}}
            ]},
            {"schema_version": 1, "status": "execute", "message": "reset", "calls": [
                {"name": "design.reset", "arguments": {}}
            ]},
        ))

        def opener(request, timeout):
            requests.append(json.loads(request.data.decode("utf-8")))
            return _Response({
                "choices": [{"message": {"content": json.dumps(next(responses))}}]
            })

        planner = OpenRouterNemotronPlanner(api_key="secret-for-test", opener=opener)
        result = planner.plan(
            instruction="Reset this design",
            current_design=self.state.to_dict(),
            capability_manifest=self.manifest,
        )

        self.assertEqual(result.status, "execute")
        self.assertEqual(len(requests), 2)
        self.assertNotIn("response_format", requests[0])
        self.assertIn("failed deterministic validation", requests[1]["messages"][-1]["content"])

    def test_openrouter_key_can_be_loaded_and_missing_key_is_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("OPENROUTER_API_KEY='file-key'\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(load_openrouter_api_key(env_file=env_file), "file-key")
                env_file.write_text("OPEN_ROUTER_API_KEY='alias-key'\n", encoding="utf-8")
                self.assertEqual(load_openrouter_api_key(env_file=env_file), "alias-key")
                planner = OpenRouterNemotronPlanner(
                    api_key="", env_file=Path(directory) / "missing.env"
                )
                with self.assertRaisesRegex(CapabilityError, "OPENROUTER_API_KEY.*Local Qwen remains available"):
                    planner.plan(
                        instruction="Make the patch wider",
                        current_design=self.state.to_dict(),
                        capability_manifest=self.manifest,
                    )

    def test_shared_exchange_is_provider_neutral(self):
        exchange = build_planner_exchange(
            instruction="Keep everything else unchanged",
            current_design=self.state.to_dict(),
            capability_manifest=self.manifest,
        )
        payload = json.loads(exchange.user_content)
        self.assertEqual(payload["instruction"], "Keep everything else unchanged")
        self.assertEqual(payload["tool_plan_schema"], exchange.schema)
        self.assertEqual(exchange.callable_names, self.names)
        self.assertEqual(
            [variant["properties"]["name"]["const"] for variant in exchange.schema["properties"]["calls"]["items"]["oneOf"]],
            list(self.names),
        )

    def test_capability_guard_blocks_small_model_substitution_and_solver_start(self):
        substituted = LLMToolPlan(
            "execute",
            "Use a dipole instead.",
            (PlannedToolCall("recipe.select", {"recipe_id": "dipole_v1"}),),
        )
        horn = enforce_request_capabilities(
            substituted,
            instruction="Replace this with a horn antenna.",
            capability_manifest=self.manifest,
        )
        solver = enforce_request_capabilities(
            substituted,
            instruction="Run the CST solver now.",
            capability_manifest=self.manifest,
        )
        self.assertEqual(horn.status, "refuse")
        self.assertEqual(horn.calls, ())
        self.assertIn("horn", horn.message)
        self.assertEqual(solver.status, "refuse")
        self.assertIn("not available", solver.message)

    def test_capability_guard_allows_a_family_when_runtime_registry_installs_it(self):
        manifest = dict(self.manifest)
        manifest["recipes"] = [*self.manifest["recipes"], {
            "recipe_id": "horn_v1",
            "family": "horn",
            "display_name": "Pyramidal horn",
            "aliases": ["horn antenna"],
        }]
        plan = LLMToolPlan("execute", "installed", (PlannedToolCall("design.reset", {}),))
        self.assertIs(
            enforce_request_capabilities(
                plan,
                instruction="Create a horn antenna.",
                capability_manifest=manifest,
            ),
            plan,
        )


if __name__ == "__main__":
    unittest.main()
