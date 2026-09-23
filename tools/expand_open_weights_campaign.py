#!/usr/bin/env python3
"""Expand the open-weights declaration into an executable generation block.

``campaigns/open_weights_protocol_v1.json`` declares that its protocol
"inherits campaigns/positional_mechanism_v1.json generation block, verbatim
except the model and sampling".  This tool makes that sentence checkable
instead of trusted: it copies the positional block, applies a closed list of
substitutions, and records both the source hash and the substitution list in
the file it writes.  Anything not in ``SUBSTITUTIONS`` or ``DROPPED`` is
inherited byte for byte, and the tool fails if the expansion it produces
disagrees with the prose the declaration already committed to.

    python tools/expand_open_weights_campaign.py            # check only
    python tools/expand_open_weights_campaign.py --write    # rewrite the file
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "campaigns" / "positional_mechanism_v1.json"
TARGET = ROOT / "campaigns" / "open_weights_protocol_v1.json"

MODEL_KEY = "llama3_1_8b"

# Everything the open-weights campaign changes about the inherited block.
SUBSTITUTIONS = {
    "output_dir": "generated_texts_open_weights_v1",
    "backend": "vllm",
    "model": {
        "key": MODEL_KEY,
        "name": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "revision": "d10aef7999a2b5ba950ab3974312feeedbfe0b77",
        "revision_note": "git revision of the HuggingFace repo, not a tag; resolved 2026-09-21 and pinned in the snapshot download",
        "mirror_note": "this is a redistributor's copy of meta-llama/Llama-3.1-8B-Instruct, not Meta's repository. Meta's repo is gated and the downloading account was not on the authorised list, so the pin rests on NousResearch rather than on Meta. The two repositories agree on the facts that can be checked without access to the gated one: 8,030,261,248 parameters, BF16, same architecture. That is consistent with identical weights but does not prove it, and the distinction is recorded here rather than glossed. Meta's own revision at the time of writing was 0e9e39f249a16976918f6564b8830bc894c89659",
        "family_note": "Llama 3.1, Meta lineage (served from a mirror); shares no training lineage with Gemini, which is the point of the campaign",
        "license": "llama3.1",
        "serving": "vLLM, local GPU",
        "dtype": "float16",
        "dtype_note": "the Avon GPU partition is Quadro RTX 6000 (Turing, compute capability 7.5), which has no bfloat16 path; the checkpoint is BF16 native and is cast to float16, which is recorded as part of the pin",
        "tensor_parallel_size": 2,
        "tensor_parallel_note": "8B at float16 is about 16 GB and would fit one 23 GB card, but that leaves roughly 6 GB for KV cache; over two cards the weights cost 8 GB each and the KV cache roughly quadruples. Llama-3.1-8B has 8 key-value heads, so 2 divides them evenly and no head is duplicated",
        "selection_note": "Qwen2.5-14B-Instruct was smoke-tested first and rejected: three of five runs continued an English prompt in Chinese (first occurrence at calls 3-8), the median continuation was about 1,000 words against the declared 1,200, and raw dash counts ran 57-106 per run against a Gemini median of 3. Llama-3.1-8B is English-centric, which addresses the first of those; the length check remains the open risk because it is a smaller model",
        "vllm_version": "<fill at freeze: recorded from the server's /version endpoint>",
        "hardware": "<fill at freeze: node, GPU model, driver>",
    },
    "conditions": {
        "c1": {
            "label": "single_prompt",
            "prompt_policy": "first_call_only",
            "runs_per_author": 10,
            "purpose": "the existing protocol: excerpt and instruction in the first call only. Generated here rather than reused, because the whole point is that both arms come from one sitting against one loaded set of weights",
        },
        "c2": {
            "label": "repeated_prompt",
            "prompt_policy": "every_call",
            "runs_per_author": 10,
            "purpose": "excerpt, author name and instruction repeated in every continuation call ahead of the same 500-word tail",
        },
    },
    "seeds_note": (
        "unlike the Gemini campaigns, this arm sets a sampling seed: the weights are local and vLLM "
        "exposes one. See sampling.seed_rule for the per-call derivation. The seeded quantities are "
        "therefore the job order and every call's sampling seed; batch composition is not seeded and "
        "is recorded instead."
    ),
    "interleaving_seed": 20260921,
    "interleaving": (
        "every (condition, author, run) job is shuffled once with the seed above and served from "
        "one worker pool, so the two conditions are interleaved in time within a single sitting"
    ),
    "sampling": {
        "seed": 20260921,
        "seed_rule": (
            "the declared seed is a campaign root, not a flat per-call seed. The seed of one call is "
            "sha256('<root>|<condition>|<author_key>|<run_id>|<prompt_sha256>') truncated to 32 bits "
            "(generate_positional_mechanism.derive_seed). The condition is in the key deliberately: c1 "
            "and c2 share a byte-identical first prompt, so a key without it would make their first "
            "windows identical by construction and the P1 equality gate vacuous. Every input to the "
            "derivation is in the call record, so each seed is recomputable from the outputs alone."
        ),
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
        "truncation_note": (
            "sampling is untruncated and unpenalised on purpose. The checkpoint's generation_config.json "
            "recommends top_p 0.8, top_k 20 and repetition_penalty 1.05 for chat use; a repetition penalty "
            "acts on punctuation tokens directly and top-p truncation reshapes the tail, so either would "
            "make the measured default a property of the decoding settings rather than of the model. "
            "vLLM 0.7 does not read generation_config.json, and these values are sent explicitly so the "
            "arm does not depend on that remaining true."
        ),
        "determinism_note": (
            "vLLM output depends on batch composition as well as seed. The campaign runs one job at a "
            "time per worker with a recorded worker count, and the realised worker count and batch "
            "composition are written to the manifest. Whether a fixed seed plus a fixed worker count "
            "reproduces byte-identical text is measured by the smoke test's duplicate-run check rather "
            "than assumed."
        ),
    },
}

# Vertex-only keys that have no meaning for a local server.
DROPPED = {
    "auth": "vertex-only; this arm serves pinned weights over a local endpoint",
    "thinking_config": "Gemini-only; Qwen2.5-14B-Instruct has no separate thinking budget",
}

# Inherited keys whose prose still names Gemini or the positional campaign.
# They are kept verbatim (that is the point) and flagged here so nobody reads
# them as descriptions of this arm.
STALE_PROSE = ["continuation_length_note", "excerpt_rule", "source_books_note"]


def sha256_obj(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_generation(source: dict) -> dict:
    # `models` is what build_plan stamps onto every job, and therefore the name
    # the client asks the server for; `model.name` is what the backend checks
    # against the served model.  They must be the same string.  Deriving one
    # from the other removes the possibility of the pair disagreeing, which it
    # did once: the model block was re-pointed at Llama while `models` still
    # said Qwen, the backend's served-name check passed, and every request 404'd.
    SUBSTITUTIONS["models"] = {MODEL_KEY: SUBSTITUTIONS["model"]["name"]}

    gen = copy.deepcopy(source["generation"])
    inherited_sha = sha256_obj(gen)
    for key in DROPPED:
        gen.pop(key, None)
    gen.update(copy.deepcopy(SUBSTITUTIONS))
    gen["generator"] = "generate_positional_mechanism.py"
    gen["inherits"] = {
        "source": "campaigns/positional_mechanism_v1.json",
        "source_generation_sha256": inherited_sha,
        "substituted_keys": sorted(SUBSTITUTIONS),
        "dropped_keys": DROPPED,
        "stale_prose_keys": STALE_PROSE,
        "note": (
            "every key not listed in substituted_keys or dropped_keys is inherited byte for byte, "
            "including the 20 authors, their book_paths, target_marks_by_cohort, excerpt_words, "
            "context_tail_words, max_rounds, dash_policy, temperature and max_output_tokens. "
            "tools/expand_open_weights_campaign.py regenerates this block and fails if it drifts."
        ),
    }
    return gen


def check_against_declaration(gen: dict, target: dict) -> list[str]:
    """The prose declaration and the executable block must agree."""
    problems: list[str] = []
    proto = target["protocol"]
    for key, declared in (
        ("excerpt_words", proto["excerpt_words"]),
        ("context_tail_words", proto["context_tail_words"]),
        ("max_rounds", proto["max_rounds"]),
        ("dash_policy", proto["dash_policy"]),
    ):
        if gen[key] != declared:
            problems.append(f"protocol.{key}={declared!r} but generation.{key}={gen[key]!r}")
    if gen["target_marks_by_cohort"] != proto["target_marks_by_cohort"]:
        problems.append("protocol.target_marks_by_cohort disagrees with the generation block")
    for cond, spec in target["conditions"].items():
        got = gen["conditions"][cond]
        if got["prompt_policy"] != spec["prompt_policy"] or got["label"] != spec["label"]:
            problems.append(f"conditions.{cond} label/policy disagree with the generation block")
        if got["runs_per_author"] != spec["runs_per_author"]:
            problems.append(f"conditions.{cond}.runs_per_author disagrees with the generation block")
    if gen["temperature"] != target["sampling"]["temperature"]:
        problems.append("sampling.temperature disagrees with the inherited generation block")
    if gen["max_output_tokens"] != target["sampling"]["max_output_tokens"]:
        problems.append("sampling.max_output_tokens disagrees with the inherited generation block")
    placeholders = [
        f"model.{k}" for k, v in target["model"].items()
        if isinstance(v, str) and v.startswith("<fill") and k not in {"vllm_version", "hardware"}
    ]
    if placeholders and target.get("status") == "frozen_before_generation":
        problems.append("frozen with unfilled placeholders: " + ", ".join(placeholders))
    if gen["interleaving_seed"] != target["interleaving"]["seed"]:
        problems.append("interleaving.seed disagrees with the generation block")
    if gen["sampling"]["seed"] != target["sampling"]["seed"]:
        problems.append("sampling.seed disagrees with the generation block")
    if gen["models"] != {gen["model"]["key"]: gen["model"]["name"]}:
        problems.append(
            f"generation.models {gen['models']!r} disagrees with generation.model "
            f"{{{gen['model']['key']!r}: {gen['model']['name']!r}}}; every request would "
            "be sent for the wrong model name"
        )
    n_runs = sum(
        spec["runs_per_author"] * len(gen["authors"]) for spec in gen["conditions"].values()
    )
    if n_runs != 400:
        problems.append(f"allocation_note declares 400 runs; the block yields {n_runs}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Rewrite the campaign file in place.")
    args = parser.parse_args()

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    target = json.loads(TARGET.read_text(encoding="utf-8"))
    gen = build_generation(source)

    problems = check_against_declaration(gen, target)
    if problems:
        raise SystemExit("declaration and generation block disagree:\n  " + "\n  ".join(problems))

    if target.get("status") == "frozen_before_generation" and target.get("generation") != gen:
        raise SystemExit("campaign is frozen and the expansion would change it; refusing")

    if not args.write:
        same = target.get("generation") == gen
        print(f"expansion {'matches' if same else 'DIFFERS FROM'} the committed generation block")
        print(f"authors={len(gen['authors'])} runs={sum(s['runs_per_author'] for s in gen['conditions'].values()) * len(gen['authors'])}")
        raise SystemExit(0 if same else 1)

    # The declaration's model/sampling blocks are projections of the executable
    # block, not a second copy to be kept in step by hand.
    target["model"] = copy.deepcopy(gen["model"])
    target["sampling"] = {
        "temperature": gen["temperature"],
        "max_output_tokens": gen["max_output_tokens"],
        **copy.deepcopy(gen["sampling"]),
    }

    # The generator names a condition directory <model_key>_<condition>, so the
    # declared output paths are projected from the block rather than written by
    # hand against an assumed layout.
    arms = ",".join(f"{MODEL_KEY}_{c}" for c in sorted(gen["conditions"]))
    target["outputs"]["generated_texts"] = f"{gen['output_dir']}/{{{arms}}}"

    ordered = {}
    for key, value in target.items():
        if key == "generation":
            continue  # rebuilt below; never carried over from the previous write
        ordered[key] = value
        if key == "interleaving":
            ordered["generation"] = gen
    if "generation" not in ordered:
        ordered["generation"] = gen
    TARGET.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}  generation_sha256={sha256_obj(gen)}")


if __name__ == "__main__":
    main()
