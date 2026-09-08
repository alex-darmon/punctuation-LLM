#!/usr/bin/env python3
"""Unit checks for segmentation, fitting and stochastic process models."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from punctlib.features import K
from punctlib.simulation import (
    DIALOGUE,
    NARRATION,
    STATE_NAMES,
    estimated_profiles,
    fit_process,
    oracle_profiles,
    score_sequence,
    segment_text,
    simulate_hsmm,
    simulate_iid,
    simulate_markov,
    stationary_distribution,
    state_profiles,
)
from run_process_simulation import (
    assert_preregistered_targets,
    chunk_sampling_frame,
    fitted_parameter_rows,
    load_fitted_parameters,
    raw_delta_gamma_diagnostic,
    validate_execution_mode,
    validation_result,
    validation_sample_rows,
)


class SegmentationTests(unittest.TestCase):
    def test_quote_marks_and_contents_are_dialogue(self) -> None:
        segmented = segment_text(
            'He paused, "Really?" Then he left.',
            strip_gutenberg=False,
        )
        self.assertEqual(segmented.marks, (",", '"', "?", '"', "."))
        self.assertEqual(
            segmented.states,
            (NARRATION, DIALOGUE, DIALOGUE, DIALOGUE, NARRATION),
        )

    def test_unclosed_quote_resets_at_paragraph_boundary(self) -> None:
        segmented = segment_text(
            '"Still speaking.\n\nNarration resumes.',
            strip_gutenberg=False,
        )
        self.assertEqual(segmented.states[-1], NARRATION)
        self.assertEqual(segmented.paragraph_unclosed_quotes, 1)

    def test_nested_directional_quotes_use_a_stack(self) -> None:
        segmented = segment_text(
            "\u201cShe called it \u2018odd\u2019!\u201d End.",
            strip_gutenberg=False,
        )
        self.assertTrue(all(state == DIALOGUE for state in segmented.states[:-1]))
        self.assertEqual(segmented.states[-1], NARRATION)
        self.assertEqual(segmented.unmatched_closers, 0)


class ProcessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.books = [
            segment_text(
                'Narration, here. "Speech!" More narration; yes.',
                strip_gutenberg=False,
            ),
            segment_text(
                'Another page: "Question?" Answer.',
                strip_gutenberg=False,
            ),
        ]
        cls.fitted = fit_process(cls.books, smoothing_eps=0.5)

    def test_fitted_probabilities_are_normalised_and_positive(self) -> None:
        self.assertAlmostEqual(float(self.fitted.marginal.sum()), 1.0)
        np.testing.assert_allclose(self.fitted.transition.sum(axis=1), 1.0)
        self.assertTrue(np.all(self.fitted.transition > 0))
        for transition in self.fitted.state_transitions:
            np.testing.assert_allclose(transition.sum(axis=1), 1.0)

    def test_contrast_zero_collapses_both_states_to_pooled(self) -> None:
        marginals, transitions = state_profiles(self.fitted, 0.0)
        for marginal in marginals:
            np.testing.assert_allclose(marginal, self.fitted.marginal)
        for transition in transitions:
            np.testing.assert_allclose(transition, self.fitted.transition)

    def test_contrast_one_returns_fitted_state_profiles(self) -> None:
        marginals, transitions = state_profiles(self.fitted, 1.0)
        for actual, expected in zip(marginals, self.fitted.state_marginals):
            np.testing.assert_allclose(actual, expected)
        for actual, expected in zip(transitions, self.fitted.state_transitions):
            np.testing.assert_allclose(actual, expected)

    def test_simulators_emit_requested_number_of_known_marks(self) -> None:
        for simulator in (
            lambda rng: simulate_iid(200, self.fitted.marginal, rng),
            lambda rng: simulate_markov(
                200, self.fitted.transition, self.fitted.marginal, rng
            ),
            lambda rng: simulate_hsmm(
                200,
                self.fitted,
                dwell_scale=1.5,
                contrast=1.0,
                rng=rng,
            ),
        ):
            sequence = simulator(np.random.default_rng(123))
            self.assertEqual(len(sequence), 200)
            self.assertTrue(all(mark in set('!"(),.:;?^') for mark in sequence))

    def test_oracle_iid_has_identical_transition_rows(self) -> None:
        marginal, transition = oracle_profiles("iid", self.fitted)
        self.assertEqual(transition.shape, (K, K))
        for row in transition:
            np.testing.assert_allclose(row, marginal)

    def test_markov_oracle_uses_the_generating_stationary_marginal(self) -> None:
        marginal, transition = oracle_profiles("markov", self.fitted)
        np.testing.assert_allclose(marginal @ transition, marginal, atol=1e-12)
        np.testing.assert_allclose(
            marginal, stationary_distribution(self.fitted.transition)
        )

    def test_estimated_profiles_pool_two_reference_sequences(self) -> None:
        marginal, transition = estimated_profiles(
            [self.books[0].marks, self.books[1].marks], smoothing_eps=0.5
        )
        self.assertAlmostEqual(float(marginal.sum()), 1.0)
        np.testing.assert_allclose(transition.sum(axis=1), 1.0)

    def test_excess_delta_subtracts_the_nominal_degrees_of_freedom(self) -> None:
        profiles = oracle_profiles("iid", self.fitted)
        sequence = simulate_iid(
            200, self.fitted.marginal, np.random.default_rng(456)
        )
        scores = score_sequence(sequence, profiles)
        self.assertAlmostEqual(
            scores["excess_delta_f3"],
            (scores["g_f3"] - K * (K - 1)) / (2 * len(sequence)),
        )

    def test_raw_delta_gamma_diagnostic_requires_positive_support(self) -> None:
        positive = raw_delta_gamma_diagnostic(
            np.asarray([0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34])
        )
        self.assertGreater(positive["raw_delta_gamma_ks_d"], 0)
        invalid = raw_delta_gamma_diagnostic(
            np.asarray([-0.01, 0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.21])
        )
        self.assertTrue(np.isnan(invalid["raw_delta_gamma_ks_d"]))

    def test_chunk_frame_weights_books_by_their_complete_chunks(self) -> None:
        designs = [
            {"target": "long", "target_length": 4500},
            {"target": "short", "target_length": 2100},
        ]
        frame = chunk_sampling_frame(designs, 1000)
        self.assertEqual(len(frame), 6)
        self.assertEqual(
            [row[0]["target"] for row in frame].count("long"), 4
        )
        self.assertEqual(
            [row[0]["target"] for row in frame].count("short"), 2
        )

    def test_fitted_parameter_artifact_round_trip(self) -> None:
        _, folds = fitted_parameter_rows({0: self.fitted})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parameters.json"
            path.write_text(json.dumps({"folds": folds}), encoding="utf-8")
            restored = load_fitted_parameters(path)[0]
        np.testing.assert_allclose(restored.transition, self.fitted.transition)
        self.assertEqual(restored.state_block_counts, self.fitted.state_block_counts)


class ValidationGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.book = segment_text(
            'Narration. "Dialogue!" End.',
            strip_gutenberg=False,
        )
        self.books = {"author": {"book.txt": self.book}}
        self.validation = {
            "sample_path": "",
            "seed": 42,
            "sample_size": 2,
            "labels": ["narration", "dialogue"],
            "max_balanced_error_rate": 0.0,
            "max_per_state_error_rate": 0.0,
        }

    @staticmethod
    def write_rows(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_validation_reconstructs_membership_and_fails_on_empty_file(self) -> None:
        root = Path(__file__).resolve().parent.parent / "results" / "repro_check"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "sample.csv"
            self.validation["sample_path"] = str(path)
            path.write_text(
                "sample_id,author,book,mark_index,mark,context,manual_state,notes\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validation_result(
                    self.validation, self.books, skip_gate=False
                )

    def test_validation_accepts_only_the_seeded_untampered_sample(self) -> None:
        root = Path(__file__).resolve().parent.parent / "results" / "repro_check"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            path = Path(directory) / "sample.csv"
            self.validation["sample_path"] = str(path)
            rows = validation_sample_rows(self.books, self.validation)
            for row in rows:
                state = self.book.states[int(row["mark_index"])]
                row["manual_state"] = STATE_NAMES[state]
            self.write_rows(path, rows)
            result = validation_result(
                self.validation, self.books, skip_gate=False
            )
            self.assertEqual(result["status"], "complete")
            rows[0]["mark"] = "!"
            self.write_rows(path, rows)
            with self.assertRaises(ValueError):
                validation_result(
                    self.validation, self.books, skip_gate=False
                )


class ExecutionGuardTests(unittest.TestCase):
    def test_smoke_cannot_target_canonical_results(self) -> None:
        root = Path(__file__).resolve().parent.parent
        config = {
            "output_dir": "results/author_panel_20/process_simulation_v1",
            "pre_registration": {"status": "draft_ready_to_freeze_before_full_run"},
        }
        args = SimpleNamespace(
            prepare_validation_sample=False,
            fit_parameters=False,
            engineering_smoke=True,
            output_dir=config["output_dir"],
            replicates=1,
            tail_replicates=1,
            reference_length_cap=100,
        )
        with self.assertRaises(ValueError):
            validate_execution_mode(
                args, config, root / config["output_dir"]
            )

    def test_preregistered_targets_include_phi_rho_and_delta(self) -> None:
        phi = {
            1000: 1.5118478780851468,
            2000: 2.2402522241351823,
            4000: 3.5552584205001576,
        }
        summaries = [
            {
                "phase": "observed",
                "feature": "f3",
                "chunk_size": size,
                "phi": value,
            }
            for size, value in phi.items()
        ]
        config = {
            "pre_registration": {
                "primary_targets": {
                    "phi": {str(size): value for size, value in phi.items()},
                    "fixed_intercept_rho": 0.0006292088575407687,
                    "implied_delta_df90": 0.02831439858933459,
                }
            }
        }
        assert_preregistered_targets(summaries, config)
        config["pre_registration"]["primary_targets"][
            "fixed_intercept_rho"
        ] += 1e-6
        with self.assertRaises(AssertionError):
            assert_preregistered_targets(summaries, config)

    def test_inferential_run_rejects_runtime_overrides(self) -> None:
        root = Path(__file__).resolve().parent.parent
        config = {
            "output_dir": "results/author_panel_20/process_simulation_v1",
            "pre_registration": {"status": "frozen_before_simulation"},
        }
        args = SimpleNamespace(
            prepare_validation_sample=False,
            fit_parameters=False,
            engineering_smoke=False,
            output_dir=None,
            replicates=10,
            tail_replicates=None,
            reference_length_cap=None,
        )
        with self.assertRaises(ValueError):
            validate_execution_mode(
                args, config, root / config["output_dir"]
            )


if __name__ == "__main__":
    unittest.main()
