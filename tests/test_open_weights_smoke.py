"""The smoke test's own logic, exercised end to end against a fake server.

The real thing costs a GPU allocation, so the parts that are ordinary
programming - selection, the checks, the determinism comparison, the files it
writes - are checked here instead.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import generate_positional_mechanism as gpm

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("ows", ROOT / "tools" / "open_weights_smoke.py")
ows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ows)

SENTENCE = "The evening closed, and she walked on, thinking; it was, she felt, enough. "


class ScriptedBackend(gpm.Backend):
    """Returns prose that depends on the seed, as a sampler would."""

    label = "scripted"
    inter_call_delay = 0.0

    def __init__(self, deterministic: bool = True) -> None:
        self.deterministic = deterministic
        self.seeds: list[int | None] = []
        self._counter = 0

    def generate(self, model, prompt, *, seed=None):
        self.seeds.append(seed)
        self._counter += 1
        salt = seed if self.deterministic else self._counter
        return gpm.CallResult(
            text=f"Chapter {salt}. " + SENTENCE * 12,
            model_version="fake@" + "0" * 40,
            response_id=f"r{self._counter}",
            finish_reason="stop",
            usage={"prompt_token_count": 1, "candidates_token_count": 2,
                   "thoughts_token_count": None, "total_token_count": 3},
        )


class TestSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.authors = json.loads(
            (ROOT / "campaigns" / "open_weights_protocol_v1.json").read_text()
        )["generation"]["authors"]

    def test_picks_five_distinct_authors_balanced_over_cohort(self) -> None:
        picked = ows.select_authors(self.authors, 5)
        self.assertEqual(len(picked), 5)
        self.assertEqual(len({a["key"] for a in picked}), 5)
        cohorts = [a["cohort"] for a in picked]
        self.assertLessEqual(abs(cohorts.count("existing") - cohorts.count("new")), 1)

    def test_selection_is_fixed(self) -> None:
        self.assertEqual(
            [a["key"] for a in ows.select_authors(self.authors, 5)],
            [a["key"] for a in ows.select_authors(self.authors, 5)],
        )


class TestRepeatDetection(unittest.TestCase):
    def test_finds_a_repeated_span(self) -> None:
        text = "unique opening. " + "A" * 250 + " middle. " + "A" * 250
        self.assertIsNotNone(ows.longest_repeated_span(text, 200))

    def test_clean_prose_has_no_repeat(self) -> None:
        varied = " ".join(f"Sentence number {i} differs from the rest." for i in range(400))
        self.assertIsNone(ows.longest_repeated_span(varied, 200))

    def test_repeat_detection_ignores_whitespace_shape(self) -> None:
        span = "B" * 220
        self.assertIsNotNone(ows.longest_repeated_span(f"{span}\n\n  {span}", 200))


class TestRunnerEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        self.config = gpm.load_config(ROOT / "campaigns" / "open_weights_protocol_v1.json")
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _job(self, target: int = 60) -> gpm.Job:
        plan = gpm.build_plan(self.config)
        base = next(j for j in plan if j.condition == "c2")
        return gpm.Job(**{**asdict(base), "target_marks": target})

    def test_run_writes_text_calls_and_a_summary_row(self) -> None:
        runner = ows.Runner(self.config, ScriptedBackend(), self.out)
        row = runner.run(self._job(), "compliance")
        self.assertGreaterEqual(row["marks"], 60)
        self.assertTrue(row["reached_target"])
        self.assertEqual(len(row["raw_sha256"]), 64)
        author = row["author_key"]
        self.assertTrue((self.out / "compliance" / author / "run_01.txt").is_file())
        calls = (self.out / "compliance" / author / "calls" / "run_01.jsonl").read_text()
        self.assertTrue(all(json.loads(line) for line in calls.splitlines() if line.strip()))

    def test_seeds_are_derived_per_call_and_recomputable(self) -> None:
        backend = ScriptedBackend()
        runner = ows.Runner(self.config, backend, self.out)
        job = self._job()
        runner.run(job, "compliance")
        self.assertTrue(all(s is not None for s in backend.seeds))
        calls = [
            json.loads(line)
            for line in (self.out / "compliance" / job.author_key / "calls" / "run_01.jsonl")
            .read_text().splitlines() if line.strip()
        ]
        recomputed = [
            gpm.derive_seed(int(self.config["generation"]["sampling"]["seed"]), job, c["prompt_sha256"])
            for c in calls
        ]
        self.assertEqual(backend.seeds, recomputed)

    def test_identical_seeds_reproduce_identical_text(self) -> None:
        """What the determinism phase compares, with the sampler held fixed."""
        job = self._job()
        a = ows.Runner(self.config, ScriptedBackend(deterministic=True), self.out / "a").run(job, "d1")
        b = ows.Runner(self.config, ScriptedBackend(deterministic=True), self.out / "b").run(job, "d2")
        self.assertEqual(a["raw_sha256"], b["raw_sha256"])

    def test_a_nondeterministic_sampler_is_detected(self) -> None:
        """One backend instance across both runs: a fresh fake would restart
        its counter and look deterministic when it is not."""
        job = self._job()
        backend = ScriptedBackend(deterministic=False)
        a = ows.Runner(self.config, backend, self.out / "c").run(job, "d1")
        b = ows.Runner(self.config, backend, self.out / "d").run(job, "d2")
        self.assertNotEqual(a["raw_sha256"], b["raw_sha256"])


class TestDashBaseline(unittest.TestCase):
    def test_median_is_read_from_the_gemini_summaries(self) -> None:
        median = ows.gemini_dash_median(ROOT / "generated_texts_positional_mechanism_v1")
        if median is None:
            self.skipTest("Gemini run tree not present")
        self.assertGreaterEqual(median, 0)

    def test_missing_tree_gives_none_so_the_caller_can_fall_back(self) -> None:
        self.assertIsNone(ows.gemini_dash_median(ROOT / "does_not_exist"))


if __name__ == "__main__":
    unittest.main()


class TestLanguageCheck(unittest.TestCase):
    """Qwen2.5 continues English prompts in Chinese; the declared checks miss it."""

    def test_english_prose_scores_zero(self) -> None:
        self.assertEqual(ows.cjk_fraction("The evening closed, and she walked on."), 0.0)

    def test_chinese_prose_scores_one(self) -> None:
        self.assertAlmostEqual(ows.cjk_fraction("随着夜幕的降临，小城沉浸在一片祥和之中。"), 1.0)

    def test_full_width_punctuation_counts(self) -> None:
        self.assertGreater(ows.cjk_fraction("　，。！？"), 0.9)

    def test_a_quoted_foreign_word_is_below_threshold(self) -> None:
        text = "He bowed and said 你好, then returned to his book by the fire. " * 4
        self.assertLess(ows.cjk_fraction(text), ows.CJK_CALL_THRESHOLD)

    def test_empty_text_is_safe(self) -> None:
        self.assertEqual(ows.cjk_fraction(""), 0.0)


class TestFailedRunsCountAsFailures(unittest.TestCase):
    """A run that died did not reach its target.

    The first version of the smoke test filtered failed runs out before
    computing the checks, so the two runs that crashed on a context overflow
    left 'reaches_target_within_max_rounds' reporting PASS.
    """

    def test_source_ties_completion_checks_to_the_failed_list(self) -> None:
        src = (ROOT / "tools" / "open_weights_smoke.py").read_text(encoding="utf-8")
        start = src.index('"reaches_target_within_max_rounds"')
        end = src.index('"median_continuation_words"', start)
        block = src[start:end]
        self.assertIn("not failed", block)
        self.assertIn("reaches_absolute_mark_floor", block)


class TestInspectorAccounting(unittest.TestCase):
    """The inspector must not double-count runs or pool the short jobs."""

    def setUp(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "owi", ROOT / "tools" / "open_weights_inspect.py")
        self.owi = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.owi)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, group: str, author: str, stem: str, n: int, words: int) -> None:
        d = self.root / group / author / "calls"
        d.mkdir(parents=True, exist_ok=True)
        recs = [
            {"call_index": i, "response_words": words, "marks_after": i * 100,
             "latency_s": 10.0, "raw_text": "x", "processed_text": "plain english text"}
            for i in range(1, n + 1)
        ]
        (d / f"{stem}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    def test_final_file_wins_over_its_live_checkpoint(self) -> None:
        self._write("compliance", "a", "run_01", 4, 1000)
        self._write("compliance", "a", "run_01.live", 2, 1000)
        paths = [p for p in self.root.rglob("*.jsonl") if p.parent.name == "calls"]
        self.assertEqual(len(paths), 2)
        by_run = {}
        for path in sorted(paths):
            stem = path.name.removesuffix(".jsonl").removesuffix(".live")
            key = (str(path.parent), stem)
            if key not in by_run or not path.name.endswith(".live.jsonl"):
                by_run[key] = path
        self.assertEqual(len(by_run), 1)
        self.assertEqual(next(iter(by_run.values())).name, "run_01.jsonl")

    def test_summarise_reads_one_run(self) -> None:
        self._write("compliance", "a", "run_01", 4, 1500)
        path = self.root / "compliance" / "a" / "calls" / "run_01.jsonl"
        row = self.owi.summarise(path)
        self.assertEqual(row["calls"], 4)
        self.assertEqual(row["median_continuation_words"], 1500)
        self.assertEqual(row["cjk_calls"], [])


class TestRepeatBaseRate(unittest.TestCase):
    """The Gemini base rate must be measured, not quoted from a one-off script."""

    def test_reports_a_rate_for_the_gemini_tree(self) -> None:
        out = ows.gemini_repeat_base_rate(
            ROOT / "generated_texts_positional_mechanism_v1", ows.REPEAT_SPAN_CHARS, sample=12)
        if out is None:
            self.skipTest("Gemini run tree not present")
        self.assertEqual(out["runs_sampled"], 12)
        self.assertGreaterEqual(out["rate"], 0.0)
        self.assertLessEqual(out["rate"], 1.0)

    def test_missing_tree_returns_none(self) -> None:
        self.assertIsNone(
            ows.gemini_repeat_base_rate(ROOT / "nope", ows.REPEAT_SPAN_CHARS))

    def test_sample_is_deterministic(self) -> None:
        a = ows.gemini_repeat_base_rate(
            ROOT / "generated_texts_positional_mechanism_v1", ows.REPEAT_SPAN_CHARS, sample=10)
        if a is None:
            self.skipTest("Gemini run tree not present")
        b = ows.gemini_repeat_base_rate(
            ROOT / "generated_texts_positional_mechanism_v1", ows.REPEAT_SPAN_CHARS, sample=10)
        self.assertEqual(a, b)
