"""The open-weights transport, and the properties the campaign depends on."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import generate_positional_mechanism as gen

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "campaigns" / "open_weights_protocol_v1.json"


def job(condition: str, author: str = "jane_austen", run_id: int = 1) -> gen.Job:
    return gen.Job(
        model_key="qwen2_5_14b",
        model="Qwen/Qwen2.5-14B-Instruct",
        condition=condition,
        prompt_policy="every_call" if condition == "c2" else "first_call_only",
        author_key=author,
        author_name="Jane Austen",
        form="prose",
        cohort="existing",
        run_id=run_id,
        target_marks=6000,
        source_book_index=0,
        source_book_path="full_books/jane_austen_full.txt",
    )


class TestSeedDerivation(unittest.TestCase):
    def test_seed_is_reproducible_from_recorded_fields(self) -> None:
        a = gen.derive_seed(20260921, job("c2"), "a" * 64)
        b = gen.derive_seed(20260921, job("c2"), "a" * 64)
        self.assertEqual(a, b)
        self.assertTrue(0 <= a < 2**32)

    def test_conditions_differ_on_the_byte_identical_first_prompt(self) -> None:
        """The P1 gate must not pass by construction.

        c1 and c2 share a byte-identical first call.  If the seed did not carry
        the condition, both arms' first windows would be the same text and the
        window-1 equality prediction would test nothing.
        """
        prompt_hash = "b" * 64
        self.assertNotEqual(
            gen.derive_seed(20260921, job("c1"), prompt_hash),
            gen.derive_seed(20260921, job("c2"), prompt_hash),
        )

    def test_runs_authors_and_calls_all_separate(self) -> None:
        seeds = {
            gen.derive_seed(20260921, job("c2", author=a, run_id=r), h)
            for a in ("jane_austen", "charles_dickens")
            for r in (1, 2)
            for h in ("c" * 64, "d" * 64)
        }
        self.assertEqual(len(seeds), 8)


class FakeCompletions:
    def __init__(self, outer: "FakeClient") -> None:
        self.outer = outer

    def create(self, **kwargs):
        self.outer.calls.append(kwargs)
        return SimpleNamespace(
            id="cmpl-1",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="Some prose, with marks."),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22, total_tokens=33),
        )


class FakeClient:
    def __init__(self, *_args, **_kwargs) -> None:
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=FakeCompletions(self))
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="Qwen/Qwen2.5-14B-Instruct")])
        )


class TestVLLMBackend(unittest.TestCase):
    def setUp(self) -> None:
        self.gen_cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["generation"]

    def _backend(self) -> tuple[gen.VLLMBackend, FakeClient]:
        import openai

        original = openai.OpenAI
        openai.OpenAI = FakeClient  # type: ignore[assignment]
        try:
            backend = gen.VLLMBackend(
                self.gen_cfg,
                base_url="http://127.0.0.1:8000/v1",
                model="Qwen/Qwen2.5-14B-Instruct",
                revision="0" * 40,
            )
        finally:
            openai.OpenAI = original  # type: ignore[assignment]
        return backend, backend.client  # type: ignore[return-value]

    def test_sampling_is_untruncated_and_unpenalised(self) -> None:
        """Truncation or a repetition penalty would reshape what is measured."""
        backend, client = self._backend()
        backend.generate("Qwen/Qwen2.5-14B-Instruct", "PROMPT", seed=7)
        sent = client.calls[0]
        self.assertEqual(sent["temperature"], 1.0)
        self.assertEqual(sent["top_p"], 1.0)
        self.assertEqual(sent["extra_body"]["top_k"], -1)
        self.assertEqual(sent["extra_body"]["repetition_penalty"], 1.0)
        self.assertEqual(sent["max_tokens"], 8192)
        self.assertEqual(sent["seed"], 7)

    def test_prompt_is_sent_verbatim_as_a_lone_user_message(self) -> None:
        backend, client = self._backend()
        backend.generate("Qwen/Qwen2.5-14B-Instruct", "PROMPT BODY", seed=1)
        self.assertEqual(
            client.calls[0]["messages"], [{"role": "user", "content": "PROMPT BODY"}]
        )

    def test_revision_travels_on_every_call(self) -> None:
        backend, _ = self._backend()
        result = backend.generate("Qwen/Qwen2.5-14B-Instruct", "P", seed=1)
        self.assertEqual(result.model_version, "Qwen/Qwen2.5-14B-Instruct@" + "0" * 40)

    def test_usage_keys_match_the_vertex_transport(self) -> None:
        """Run summaries read these names; both transports must supply them."""
        backend, _ = self._backend()
        usage = backend.generate("Qwen/Qwen2.5-14B-Instruct", "P", seed=1).usage
        self.assertEqual(usage["prompt_token_count"], 11)
        self.assertEqual(usage["candidates_token_count"], 22)
        self.assertEqual(usage["total_token_count"], 33)
        self.assertIsNone(usage["thoughts_token_count"])


class TestBackendDispatch(unittest.TestCase):
    def test_placeholder_or_tag_revision_is_refused(self) -> None:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))["generation"]
        for bad in ("<fill: git revision hash of the weights, not a tag>", "main", "v1.0"):
            cfg["model"]["revision"] = bad
            with self.assertRaises(SystemExit):
                gen.make_backend(cfg, SimpleNamespace(base_url="http://x/v1"))

    def test_unknown_backend_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            gen.make_backend({"backend": "nonsense"}, SimpleNamespace(base_url=None))


class TestCampaignConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.config = gen.load_config(CONFIG)

    def test_plan_is_400_runs_split_evenly(self) -> None:
        plan = gen.build_plan(self.config)
        self.assertEqual(len(plan), 400)
        self.assertEqual(sum(1 for j in plan if j.condition == "c1"), 200)
        self.assertEqual(sum(1 for j in plan if j.condition == "c2"), 200)

    def test_conditions_are_interleaved_in_one_batch(self) -> None:
        order = gen.interleave(
            gen.build_plan(self.config), self.config["generation"]["interleaving_seed"]
        )
        first_fifty = {j.condition for j in order[:50]}
        self.assertEqual(first_fifty, {"c1", "c2"})

    def test_protocol_is_the_gemini_protocol(self) -> None:
        """The whole point of reusing the generator: same prompts, same hash."""
        self.assertEqual(gen.legacy_protocol_sha256(), gen.LEGACY_PROTOCOL_SHA256)

    def test_declaration_and_generation_block_agree(self) -> None:
        raw = json.loads(CONFIG.read_text(encoding="utf-8"))
        block = raw["generation"]
        self.assertEqual(block["temperature"], raw["sampling"]["temperature"])
        self.assertEqual(block["max_output_tokens"], raw["sampling"]["max_output_tokens"])
        self.assertEqual(block["sampling"]["seed"], raw["sampling"]["seed"])
        self.assertEqual(block["interleaving_seed"], raw["interleaving"]["seed"])
        self.assertEqual(block["model"]["name"], raw["model"]["name"])
        self.assertEqual(len(block["model"]["revision"]), 40)


class TestClientEnvironment(unittest.TestCase):
    def test_records_the_packages_that_define_the_stopping_rule(self) -> None:
        env = gen.client_environment()
        for key in ("spacy", "en_core_web_sm", "numpy", "python"):
            self.assertIn(key, env)


if __name__ == "__main__":
    unittest.main()


class TestModelNameConsistency(unittest.TestCase):
    """`generation.models` and `generation.model.name` must be the same string.

    build_plan stamps `models[key]` onto every Job, so that is the name the
    client sends; VLLMBackend validates `model.name` against the served model.
    When a model swap updated only the second, the backend started cleanly and
    every single request came back 404.
    """

    def setUp(self) -> None:
        self.raw = json.loads(CONFIG.read_text(encoding="utf-8"))["generation"]

    def test_models_dict_matches_the_model_block(self) -> None:
        self.assertEqual(
            self.raw["models"],
            {self.raw["model"]["key"]: self.raw["model"]["name"]},
        )

    def test_every_job_carries_the_served_model_name(self) -> None:
        plan = gen.build_plan({"generation": self.raw})
        self.assertEqual({j.model for j in plan}, {self.raw["model"]["name"]})

    def test_expander_rejects_a_divergent_models_dict(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "exp", ROOT / "tools" / "expand_open_weights_campaign.py")
        exp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exp)
        gen_block = dict(self.raw)
        gen_block["models"] = {self.raw["model"]["key"]: "some/other-model"}
        target = json.loads(CONFIG.read_text(encoding="utf-8"))
        problems = exp.check_against_declaration(gen_block, target)
        self.assertTrue(any("disagrees with generation.model" in p for p in problems), problems)
