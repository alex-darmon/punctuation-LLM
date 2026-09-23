#!/usr/bin/env python3
"""Positional-mechanism campaign: prompt assembly, boundaries, plan, analysis on synthetic runs."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

import generate_positional_mechanism as gen
import run_positional_mechanism as ana

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "campaigns" / "positional_mechanism_v1.json"
RESULTS = ROOT / "results" / "author_panel_20" / "positional_mechanism_v1"


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def word_counter(text: str) -> int:
    """Stand-in mark counter: one mark per word, so lengths are exact in tests."""
    return len(text.split())


class MockBackend:
    """Deterministic generator that records every prompt it receives."""

    def __init__(self, words_per_call: int = 40) -> None:
        self.prompts: list[str] = []
        self.words_per_call = words_per_call

    def generate(self, model: str, prompt: str) -> gen.CallResult:
        self.prompts.append(prompt)
        n = len(self.prompts)
        text = " ".join(f"w{n}_{i}" for i in range(self.words_per_call))
        return gen.CallResult(
            text=text,
            model_version=f"{model}-mock",
            response_id=f"r{n}",
            finish_reason="STOP",
            usage={"prompt_token_count": 10, "candidates_token_count": 20, "thoughts_token_count": 0, "total_token_count": 30},
        )


def make_job(condition: str, policy: str, target: int = 100) -> gen.Job:
    return gen.Job(
        model_key="flash", model="gemini-mock", condition=condition, prompt_policy=policy,
        author_key="a", author_name="Author A", form="prose", cohort="new", run_id=1,
        target_marks=target, source_book_index=0, source_book_path="x.txt",
    )


class TemplateTests(unittest.TestCase):
    def test_legacy_templates_match_extension_batches(self) -> None:
        self.assertEqual(gen.legacy_protocol_sha256(), gen.LEGACY_PROTOCOL_SHA256)
        for directory in ("generated_texts_campaign_author20_flash_new10", "generated_texts_campaign_author20_pro_new10"):
            path = ROOT / directory / "campaign_metadata.json"
            if path.is_file():
                recorded = json.loads(path.read_text(encoding="utf-8"))["prompt_protocol_sha256"]
                self.assertEqual(recorded, gen.LEGACY_PROTOCOL_SHA256)

    def test_c2_continuation_contains_excerpt_author_and_tail(self) -> None:
        prompt, kind, present = gen.build_prompt(
            call_index=2, prompt_policy="every_call", form="prose", author_name="Jane Doe",
            excerpt="EXCERPT_TOKEN", accumulated_text="one two three", context_tail_words=2,
        )
        self.assertEqual(kind, "continuation_with_prompt")
        self.assertTrue(present)
        self.assertIn("EXCERPT_TOKEN", prompt)
        self.assertIn("Jane Doe", prompt)
        self.assertIn("two three", prompt)
        self.assertNotIn("one two three", prompt)
        self.assertIn("Write at least 2000 words", prompt)

    def test_c1_continuation_is_the_legacy_template(self) -> None:
        prompt, kind, present = gen.build_prompt(
            call_index=2, prompt_policy="first_call_only", form="prose", author_name="Jane Doe",
            excerpt="EXCERPT_TOKEN", accumulated_text="one two three", context_tail_words=500,
        )
        self.assertEqual(kind, "continuation")
        self.assertFalse(present)
        self.assertEqual(prompt, gen.PROSE_CONTINUATION_PROMPT.format(last_section="one two three"))
        self.assertNotIn("EXCERPT_TOKEN", prompt)
        self.assertNotIn("Jane Doe", prompt)

    def test_first_call_identical_across_arms(self) -> None:
        kwargs = dict(call_index=1, form="prose", author_name="Jane Doe", excerpt="E", accumulated_text="", context_tail_words=500)
        c1 = gen.build_prompt(prompt_policy="first_call_only", **kwargs)
        c2 = gen.build_prompt(prompt_policy="every_call", **kwargs)
        self.assertEqual(c1, c2)
        self.assertEqual(c1[0], gen.PROSE_PROMPT.format(author_name="Jane Doe", excerpt="E"))


class MockGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gen_cfg = {"context_tail_words": 5, "max_rounds": 10, "dash_policy": "replace_with_comma"}

    def _run(self, policy: str, condition: str):
        backend = MockBackend(words_per_call=40)
        job = make_job(condition, policy, target=100)
        text, raw, calls = gen.generate_run(
            job, gen=self.gen_cfg, excerpt="EXCERPT_TOKEN", generate=backend.generate,
            mark_counter=word_counter, sleep=lambda s: None, log=lambda m: None,
        )
        return backend, text, calls

    def test_c1_prompt_only_in_first_call(self) -> None:
        backend, text, calls = self._run("first_call_only", "c1")
        self.assertEqual(len(calls), 3)
        self.assertIn("EXCERPT_TOKEN", backend.prompts[0])
        for prompt in backend.prompts[1:]:
            self.assertNotIn("EXCERPT_TOKEN", prompt)
        self.assertEqual([c["excerpt_present"] for c in calls], [True, False, False])
        self.assertEqual([c["prompt_kind"] for c in calls], ["initial", "continuation", "continuation"])

    def test_c2_prompt_in_every_call(self) -> None:
        backend, text, calls = self._run("every_call", "c2")
        self.assertEqual(len(calls), 3)
        for prompt in backend.prompts:
            self.assertIn("EXCERPT_TOKEN", prompt)
        self.assertTrue(all(c["excerpt_present"] for c in calls))
        self.assertEqual([c["prompt_kind"] for c in calls], ["initial", "continuation_with_prompt", "continuation_with_prompt"])

    def test_boundaries_and_tail_are_logged(self) -> None:
        backend, text, calls = self._run("every_call", "c2")
        self.assertEqual([c["marks_before"] for c in calls], [0, 40, 80])
        self.assertEqual([c["marks_after"] for c in calls], [40, 80, 120])
        self.assertEqual(calls[-1]["marks_after"], word_counter(text))
        # the continuation tail is the last five words of the accumulated text
        self.assertIn(" ".join(text.split()[35:40]), backend.prompts[1])
        self.assertEqual(calls[1]["tail_words"], 5)
        self.assertTrue(all(c["model_version"] == "gemini-mock-mock" for c in calls))
        # per-call processed texts re-assemble into the run text
        self.assertEqual("\n\n".join(c["processed_text"] for c in calls), text)

    def test_resume_continues_from_saved_calls_and_checkpoints(self) -> None:
        backend = MockBackend(words_per_call=40)
        job = make_job("c2", "every_call", target=200)
        saved: list[dict] = []
        # first attempt: stop after two calls by exhausting attempts on the third
        calls_seen = {"n": 0}

        def flaky(model: str, prompt: str) -> gen.CallResult:
            calls_seen["n"] += 1
            if calls_seen["n"] == 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return backend.generate(model, prompt)

        with self.assertRaises(RuntimeError):
            gen.generate_run(
                job, gen=self.gen_cfg, excerpt="EXCERPT_TOKEN", generate=flaky, mark_counter=word_counter,
                sleep=lambda s: None, log=lambda m: None, max_attempts_per_call=1, checkpoint=saved.append,
            )
        self.assertEqual([c["call_index"] for c in saved], [1, 2])
        # second attempt resumes after call 2 and finishes the run
        backend2 = MockBackend(words_per_call=40)
        text, raw, calls = gen.generate_run(
            job, gen=self.gen_cfg, excerpt="EXCERPT_TOKEN", generate=backend2.generate, mark_counter=word_counter,
            sleep=lambda s: None, log=lambda m: None, resume_calls=saved,
        )
        self.assertEqual([c["call_index"] for c in calls], [1, 2, 3, 4, 5])
        self.assertEqual(calls[2]["marks_before"], 80)
        self.assertEqual(word_counter(text), 200)
        self.assertEqual("\n\n".join(c["processed_text"] for c in calls), text)
        # the resumed continuation carried the tail of the saved text
        self.assertIn(" ".join(saved[1]["processed_text"].split()[-5:]), backend2.prompts[0])

    def test_billing_and_quota_errors_stop_the_batch(self) -> None:
        for message in ("429 per_model_per_day", "403 PERMISSION_DENIED BILLING_DISABLED", "requires billing to be enabled"):
            self.assertTrue(gen.is_fatal_error(message))
        self.assertFalse(gen.is_fatal_error("429 RESOURCE_EXHAUSTED"))

        def billing(model: str, prompt: str) -> gen.CallResult:
            raise RuntimeError("403 PERMISSION_DENIED. requires billing to be enabled")

        job = make_job("c2", "every_call", target=100)
        with self.assertRaises(gen.QuotaExhausted):
            gen.generate_run(job, gen=self.gen_cfg, excerpt="E", generate=billing, mark_counter=word_counter, sleep=lambda s: None, log=lambda m: None)

    def test_dash_policy_applied_per_call(self) -> None:
        backend = MockBackend()
        backend.generate = lambda model, prompt: gen.CallResult(  # type: ignore[assignment]
            text="alpha — beta – gamma " * 30, model_version="v", response_id=None, finish_reason=None, usage={}
        )
        job = make_job("c1", "first_call_only", target=50)
        text, raw, calls = gen.generate_run(
            job, gen=self.gen_cfg, excerpt="E", generate=backend.generate,
            mark_counter=word_counter, sleep=lambda s: None, log=lambda m: None,
        )
        self.assertNotIn("—", text)
        self.assertIn("—", raw)
        self.assertNotIn("—", calls[0]["processed_text"])
        self.assertIn("—", calls[0]["raw_text"])


class PlanTests(unittest.TestCase):
    def test_plan_is_balanced_and_matches_declared_counts(self) -> None:
        config = load_config()
        jobs = gen.build_plan(config)
        conditions = config["generation"]["conditions"]
        n_authors = len(config["generation"]["authors"])
        expected = n_authors * sum(
            gen.runs_per_author(spec, model)
            for spec in conditions.values()
            for model in config["generation"]["models"]
        )
        self.assertEqual(len(jobs), expected)
        # the run counts of the existing campaigns: Flash 20 and Pro 10 per author
        self.assertEqual(expected, 600)
        per_arm = {(j.model_key, j.condition) for j in jobs}
        self.assertEqual(per_arm, {("flash", "c2"), ("pro", "c2")})
        self.assertEqual(sum(j.model_key == "flash" for j in jobs), 400)
        self.assertEqual(sum(j.model_key == "pro" for j in jobs), 200)
        for job in jobs:
            self.assertIn(job.condition, conditions)
            self.assertEqual(job.prompt_policy, conditions[job.condition]["prompt_policy"])
        # every author's declared books appear in every condition, in rotation order
        for author in config["generation"]["authors"]:
            for model in config["generation"]["models"]:
                for condition in conditions:
                    used = [j.source_book_path for j in jobs if j.author_key == author["key"] and j.model_key == model and j.condition == condition]
                    self.assertEqual(used[: len(author["book_paths"])], author["book_paths"][: len(used)])
                    self.assertEqual(set(used), set(author["book_paths"][: len(used)]))

    def test_run_counts_may_be_one_number_or_one_per_model(self) -> None:
        self.assertEqual(gen.runs_per_author({"runs_per_author": 10}, "flash"), 10)
        spec = {"runs_per_author": {"flash": 20, "pro": 10}}
        self.assertEqual(gen.runs_per_author(spec, "flash"), 20)
        self.assertEqual(gen.runs_per_author(spec, "pro"), 10)

    def test_amendment_names_the_batch_config_it_replaces(self) -> None:
        # --resume continues the existing batch only because the amended file
        # declares the hash the batch was started under.
        config = load_config()
        declared = {a["previous_config_sha256"] for a in config["amendments"]}
        metadata_path = gen.ROOT / config["generation"]["output_dir"] / "campaign_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            known = {metadata["campaign_config_sha256"], *metadata.get("superseded_campaign_config_sha256", [])}
            self.assertTrue(declared & known)

    def test_interleaving_is_seeded_and_mixes_arms(self) -> None:
        config = load_config()
        jobs = gen.build_plan(config)
        seed = int(config["generation"]["interleaving_seed"])
        a = gen.interleave(jobs, seed)
        b = gen.interleave(jobs, seed)
        self.assertEqual(a, b)
        first_half = a[: len(a) // 2]
        active = {
            c
            for c, spec in config["generation"]["conditions"].items()
            if any(gen.runs_per_author(spec, m) > 0 for m in config["generation"]["models"])
        }
        self.assertEqual({j.condition for j in first_half}, active)
        self.assertTrue({j.model_key for j in first_half} == set(config["generation"]["models"]))

    def test_target_marks_follow_cohort(self) -> None:
        config = load_config()
        targets = config["generation"]["target_marks_by_cohort"]
        for author in config["generation"]["authors"]:
            self.assertEqual(author["target_marks"], targets[author["cohort"]])
        self.assertEqual(targets, {"existing": 6000, "new": 5000})

    def test_generator_refuses_unfrozen_config_for_inferential_run(self) -> None:
        config = load_config()
        self.assertIn(config["status"], {"design_review", "frozen_before_generation"})
        self.assertEqual(config["generation"]["continuation_length_policy"], "fixed_template")
        self.assertEqual(config["generation"]["auth"]["mode"], "adc")


class BoundaryTests(unittest.TestCase):
    def test_per_call_sum_is_used_when_it_matches_cache(self) -> None:
        calls = [{"processed_text": "a b c"}, {"processed_text": "d e"}]
        ends, method = ana.derive_boundaries(calls, cached_length=5, parse_length=word_counter)
        self.assertEqual(ends, [3, 5])
        self.assertEqual(method, "per_call_sum")

    def test_cumulative_parse_fallback(self) -> None:
        calls = [{"processed_text": "a b c"}, {"processed_text": "d e"}]
        # cumulative parser that finds one extra mark at a join
        ends, method = ana.derive_boundaries(calls, cached_length=6, parse_length=lambda t: len(t.split()) + t.count("\n\n"))
        self.assertEqual(ends, [3, 6])
        self.assertEqual(method, "cumulative_parse")

    def test_call_position(self) -> None:
        ends = [1200, 2600, 4100, 5300]
        self.assertEqual(ana.call_position(ends, 0), (1, 0))
        self.assertEqual(ana.call_position(ends, 1000), (1, 1000))
        self.assertEqual(ana.call_position(ends, 2000), (2, 800))
        self.assertEqual(ana.call_position(ends, 4000), (3, 1400))


class SyntheticAnalysisTests(unittest.TestCase):
    """The estimators must see drift when it is injected and nothing when it is not."""

    @staticmethod
    def rows(model: str, arm: str, slope: float, authors: int = 6, runs: int = 4, noise: float = 0.0, seed: int = 0) -> list[dict]:
        rng = np.random.default_rng(seed)
        out = []
        for a in range(authors):
            for r in range(runs):
                for w in range(1, 6):
                    kl_value = 0.2 + slope * (w - 1) + noise * rng.normal()
                    out.append({
                        "condition": arm, "author_set": "all_authors", "feature": "f3", "window_size": 1000,
                        "window_index": w, "start_mark": (w - 1) * 1000, "end_mark": w * 1000,
                        "target_author": f"author{a}", "run_id": r + 1, "source_book": "b",
                        "target_kl": kl_value, "nearest_other_kl": 0.3, "target_margin": 0.3 - kl_value,
                        "target_rank": 1 + int(kl_value > 0.3), "target_hit": kl_value <= 0.3, "predicted_author": "x",
                        "model": model, "arm": arm, "offset": 0, "source_type": "llm",
                    })
        return out

    def test_injected_drift_is_positive_and_flat_runs_are_zero(self) -> None:
        rows = self.rows("flash", "old", slope=0.05, noise=0.005) + self.rows("flash", "c2", slope=0.0, noise=0.005, seed=1)
        deltas = ana.run_deltas(rows, 5)
        estimates = ana.arm_estimates(deltas, models_arms=[("flash", "old"), ("flash", "c2")], n_boot=200, seed=1, confidence=0.95, two_sided_arms=set())
        old = next(r for r in estimates if r["arm"] == "old" and r["metric"] == "target_kl")
        c2 = next(r for r in estimates if r["arm"] == "c2" and r["metric"] == "target_kl")
        self.assertAlmostEqual(old["estimate"], 0.2, delta=0.02)
        self.assertGreater(old["ci_low"], 0.15)
        self.assertLess(old["wilcoxon_p"], 0.05)
        self.assertAlmostEqual(c2["estimate"], 0.0, delta=0.02)
        self.assertLessEqual(c2["ci_low"], 0.0)
        self.assertGreaterEqual(c2["ci_high"], 0.0)

    def test_contrast_detects_less_drift_and_bh_is_filled(self) -> None:
        rows = (
            self.rows("flash", "old", slope=0.05, noise=0.005)
            + self.rows("flash", "replicate", slope=0.05, noise=0.005, seed=2)
            + self.rows("flash", "c2", slope=0.0, noise=0.005, seed=3)
        )
        deltas = ana.run_deltas(rows, 5)
        contrasts = ana.contrast_rows(deltas, models=["flash"], n_boot=200, seed=1, confidence=0.95)
        primary = next(r for r in contrasts if r["contrast"] == "c2_minus_c1_pooled" and r["metric"] == "target_kl")
        self.assertLess(primary["ci_high"], 0.0)
        self.assertLess(float(primary["benjamini_hochberg_p"]), 0.05)
        batch = next(r for r in contrasts if r["contrast"] == "replicate_minus_old" and r["metric"] == "target_kl")
        self.assertAlmostEqual(batch["mean_difference"], 0.0, delta=0.01)
        self.assertLess(batch["ci_high"] - batch["ci_low"], 0.05)
        self.assertEqual(batch["alternative"], "two-sided")
        self.assertEqual(batch["benjamini_hochberg_p"], "")
        window1 = next(r for r in contrasts if r["contrast"] == "c2_minus_c1_pooled_window1" and r["metric"] == "target_kl")
        self.assertAlmostEqual(window1["mean_difference"], 0.0, delta=0.01)

    def test_without_replicate_runs_the_comparator_is_the_old_arm(self) -> None:
        rows = self.rows("flash", "old", slope=0.05, noise=0.005) + self.rows("flash", "c2", slope=0.0, noise=0.005, seed=3)
        deltas = ana.run_deltas(rows, 5)
        estimates = ana.arm_estimates(
            deltas, models_arms=[("flash", "old"), ("flash", "c2"), ("flash", "c1_pooled"), ("flash", "replicate")],
            n_boot=100, seed=1, confidence=0.95, two_sided_arms=set(),
        )
        self.assertFalse([r for r in estimates if r["arm"] == "replicate"])
        contrasts = ana.contrast_rows(deltas, models=["flash"], n_boot=100, seed=1, confidence=0.95)
        self.assertFalse([r for r in contrasts if r["contrast"] == "replicate_minus_old"])
        config = load_config()
        verdict = ana.evaluate_predictions(
            config=config, models=["flash"], estimates=estimates, contrasts=contrasts, share_contrasts=[],
            detection_contrasts=[], regressions=[], regression_check_result={"passed": True}, human_estimates=[],
        )
        self.assertEqual(verdict["P1_batch_check"]["flash"]["status"], "not_testable_no_replicate_runs")
        self.assertEqual(verdict["P2_prompt_fading"]["flash"]["comparator"], "c1_pooled")
        self.assertEqual(verdict["P2_prompt_fading"]["flash"]["reading"], "prompt_fading")

    def test_interior_offset_respects_bounds(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            offset = ana.interior_offset(12000, 5000, 1000, rng)
            self.assertIsNotNone(offset)
            self.assertGreaterEqual(offset, 1000)
            self.assertLessEqual(offset, 7000)
        self.assertIsNone(ana.interior_offset(5500, 5000, 1000, rng))


@unittest.skipUnless((RESULTS / "manifest.json").is_file(), "run run_positional_mechanism.py first")
class CommittedResultTests(unittest.TestCase):
    def test_manifest_hashes_outputs_and_regression_check_passed(self) -> None:
        import hashlib

        manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
        for name, digest in manifest["output_sha256"].items():
            self.assertEqual(hashlib.sha256((RESULTS / name).read_bytes()).hexdigest(), digest, name)
        check = json.loads((RESULTS / "regression_check.json").read_text(encoding="utf-8"))
        self.assertTrue(check["passed"])
        self.assertEqual(check["compared"], check["pinned_rows"])

    def test_old_arm_reproduces_pinned_drift_table(self) -> None:
        import csv

        pinned = ROOT / "results" / "author_panel_20" / "inference_v2" / "drift_clustered.csv"
        with pinned.open(newline="", encoding="utf-8") as handle:
            pinned_rows = [r for r in csv.DictReader(handle) if r["row_type"] == "delta"]
        with (RESULTS / "window_summary.csv").open(newline="", encoding="utf-8") as handle:
            ours = [r for r in csv.DictReader(handle) if r["row_type"] == "delta" and r["arm"] == "old"]
        for p in pinned_rows:
            row = next(r for r in ours if r["model"] == p["condition"] and r["feature"] == p["feature"] and r["metric"] == p["metric"])
            self.assertAlmostEqual(float(row["estimate"]), float(p["estimate"]), places=9)
            self.assertAlmostEqual(float(row["wilcoxon_p"]), float(p["wilcoxon_p"]), places=9)

    def test_human_control_covers_declared_books(self) -> None:
        import csv

        with (RESULTS / "human_control_eligibility.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 60)
        self.assertEqual(sum(r["eligible_start"] == "True" for r in rows), 55)
        self.assertNotIn("arthur_scott_bailey", {r["author"] for r in rows if r["eligible_start"] == "True"})


if __name__ == "__main__":
    unittest.main()
