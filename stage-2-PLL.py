#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-2 PPL-based verification for deleted words.

For each sample from Stage-1 output:
  1. Restore all removed punctuation symbols.
  2. Split text into sentences by ".".
  3. For sentences where exactly ONE word was removed:
     - Compute GPT-2 PPL before and after removal.
     - If removing the word significantly reduced PPL (ppl_after < ppl_before - threshold),
       keep it deleted (the word was anomalous).
     - Otherwise, restore the word (it was a likely false positive).
  4. Recompute localization and clean-sample metrics before/after stage-2.
"""

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

import torch
import transformers
from tqdm import tqdm

_PUNCT_RE = re.compile(r"^[^\w]+$")
_STRIP_PUNCT = re.compile(r"^[.,!?;:'\"()\[\]{}/]+|[.,!?;:'\"()\[\]{}/]+$")


def _strip_word(w):
    return _STRIP_PUNCT.sub("", w).lower()


def find_trigger_indices(words, trigger_str):
    """Find trigger word indices in word list.

    If trigger_str contains commas → distributed mode (exact match each token).
    Otherwise → phrase mode (contiguous subsequence match).
    """
    if "," in trigger_str:
        trig_tokens = [_strip_word(t.strip()) for t in trigger_str.split(",") if t.strip()]
        indices = set()
        for t in trig_tokens:
            for i, w in enumerate(words):
                if _strip_word(w) == t:
                    indices.add(i)
        return indices
    else:
        trig_tokens = [_strip_word(t) for t in trigger_str.split()]
        trig_tokens = [t for t in trig_tokens if t]
        tlen = len(trig_tokens)
        if tlen == 0:
            return set()
        indices = set()
        words_stripped = [_strip_word(w) for w in words]
        for start in range(len(words) - tlen + 1):
            if all(words_stripped[start + j] == trig_tokens[j] for j in range(tlen)):
                for j in range(tlen):
                    indices.add(start + j)
        return indices


def compute_poison_metrics_with_trigger(samples, trigger_str):
    """Recompute P/R/F1 from scratch using trigger string matching."""
    tp_total, fp_total, fn_total = 0, 0, 0
    count = 0
    for s in samples:
        if not s.get("is_poisoned", False):
            continue
        words = s.get("original_text", "").split()
        true_indices = find_trigger_indices(words, trigger_str)
        if not true_indices:
            continue

        spans = s.get("best_spans", [])
        deleted_indices = set()
        for st, e in spans:
            for j in range(st, e):
                deleted_indices.add(j)

        punct_only = {i for i in range(len(words))
                      if all(c in ".,!?;:'\"()-[]{}/" for c in words[i])}
        true_indices -= punct_only
        deleted_indices -= punct_only

        if not true_indices:
            continue

        count += 1
        tp_total += len(deleted_indices & true_indices)
        fp_total += len(deleted_indices - true_indices)
        fn_total += len(true_indices - deleted_indices)

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "count": count,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


# ------------------------------------------------------------------
# GPT-2 perplexity scorer
# ------------------------------------------------------------------
class GPT2PPL:
    def __init__(self, device=None):
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
        self.model.eval()
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def __call__(self, text: str) -> float:
        text = text.strip()
        if not text:
            return float("inf")
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        ).to(self.device)
        outputs = self.model(**inputs, labels=inputs.input_ids)
        return torch.exp(outputs.loss).item()


# ------------------------------------------------------------------
# Metric helpers
# ------------------------------------------------------------------
def compute_poison_metrics(samples):
    """Aggregate P / R / F1 for poisoned samples using per-sample tp/fp/fn."""
    tp_total, fp_total, fn_total = 0, 0, 0
    count = 0
    for s in samples:
        if not s.get("is_poisoned", False):
            continue
        m = s.get("localization_metrics", {})
        tp_total += m.get("tp", 0)
        fp_total += m.get("fp", 0)
        fn_total += m.get("fn", 0)
        count += 1

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "count": count,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def compute_clean_metrics(samples):
    """Average removed-word count and percentage for clean samples."""
    removed_counts = []
    removed_pcts = []
    for s in samples:
        if s.get("is_poisoned", False):
            continue
        spans = s.get("best_spans", [])
        num_del = sum(e - st for st, e in spans)
        total_words = len(s.get("original_text", "").split())
        removed_counts.append(num_del)
        removed_pcts.append(num_del / total_words * 100 if total_words > 0 else 0.0)

    n = len(removed_counts)
    return {
        "count": n,
        "avg_removed_words": round(sum(removed_counts) / n, 4) if n else 0.0,
        "avg_removed_pct": round(sum(removed_pcts) / n, 4) if n else 0.0,
    }


# ------------------------------------------------------------------
# Per-sample stage-2 restoration
# ------------------------------------------------------------------
def _assign_sentences(words):
    """Map each word index to a sentence id (sentences delimited by '.')."""
    sent_id = 0
    mapping = []
    for w in words:
        mapping.append(sent_id)
        if "." in w:
            sent_id += 1
    return mapping


_PAD_SENTENCE = "This is just an ordinary sentence, used to occupy space."
_MIN_WORDS = 40


def _pad_short_sentence(text):
    """If text has fewer than _MIN_WORDS words, pad by repeating filler."""
    ws = text.split()
    while len(ws) < _MIN_WORDS:
        ws.extend(_PAD_SENTENCE.split())
    return " ".join(ws)


def process_sample(sample, ppl_model, ppl_threshold):
    """Apply stage-2 PPL-based restoration to one sample.

    Returns a new sample dict with updated spans / text / metrics.
    """
    original_text = sample.get("original_text", "")
    words = original_text.split()
    spans = sample.get("best_spans", [])

    removed_indices = set()
    for start, end in spans:
        for i in range(start, end):
            if i < len(words):
                removed_indices.add(i)

    if not removed_indices:
        return dict(sample)

    # --- Step 1: restore punctuation ---
    punct_restored = {
        idx for idx in removed_indices
        if idx < len(words) and _PUNCT_RE.match(words[idx])
    }
    remaining_removed = removed_indices - punct_restored

    if not remaining_removed:
        return _rebuild_sample(sample, words, set())

    # --- Step 2: sentence assignment ---
    word_to_sent = _assign_sentences(words)

    removed_by_sent = defaultdict(list)
    for idx in remaining_removed:
        removed_by_sent[word_to_sent[idx]].append(idx)

    words_by_sent = defaultdict(list)
    for i in range(len(words)):
        words_by_sent[word_to_sent[i]].append(i)

    # --- Step 3: PPL check for single-removal sentences ---
    ppl_restored = set()
    for sent_id, removed_list in removed_by_sent.items():
        if len(removed_list) != 1:
            continue

        removed_idx = removed_list[0]
        sent_indices = words_by_sent[sent_id]

        sentence_before = " ".join(words[i] for i in sent_indices)
        sentence_after = " ".join(words[i] for i in sent_indices if i != removed_idx)

        if not sentence_before.strip() or not sentence_after.strip():
            continue

        sentence_before = _pad_short_sentence(sentence_before)
        sentence_after = _pad_short_sentence(sentence_after)

        ppl_before = ppl_model(sentence_before)
        ppl_after = ppl_model(sentence_after)

        is_poisoned = sample.get("is_poisoned", False)
        delta = ppl_after - ppl_before
        action = "RESTORE" if delta > ppl_threshold else "KEEP_DEL"
        print(f"  word='{words[removed_idx]}' delta={delta:.4f} is_poisoned={is_poisoned} → {action}")

        if delta > ppl_threshold:
            ppl_restored.add(removed_idx)

    final_removed = remaining_removed - ppl_restored
    total_restored = len(punct_restored) + len(ppl_restored)

    new_sample = _rebuild_sample(sample, words, final_removed)

    # --- Update per-sample localization_metrics if present ---
    if "localization_metrics" in sample and sample.get("is_poisoned", False):
        old = sample["localization_metrics"]
        old_fp = old.get("fp", 0)
        old_tp = old.get("tp", 0)
        old_fn = old.get("fn", 0)

        fp_reduction = min(total_restored, old_fp)
        tp_reduction = total_restored - fp_reduction

        new_sample["localization_metrics"] = {
            "tp": max(old_tp - tp_reduction, 0),
            "fp": max(old_fp - fp_reduction, 0),
            "fn": old_fn + tp_reduction,
        }

    return new_sample


def _rebuild_sample(sample, words, final_removed):
    """Build a new sample dict with updated spans, removed text, and ablated text."""
    new_spans = []
    for idx in sorted(final_removed):
        if new_spans and new_spans[-1][1] == idx:
            new_spans[-1][1] = idx + 1
        else:
            new_spans.append([idx, idx + 1])

    new_removed_text = " ".join(words[i] for i in sorted(final_removed))
    new_ablated = " ".join(words[i] for i in range(len(words)) if i not in final_removed)

    ns = dict(sample)
    ns["best_spans"] = new_spans
    ns["best_removed_text"] = new_removed_text
    ns["ablated_text_raw"] = new_ablated
    return ns


# ------------------------------------------------------------------
# File-level processing
# ------------------------------------------------------------------
def process_file(input_path, output_path, ppl_model, ppl_threshold, trigger=None):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        samples = data.get("samples", [])
    else:
        samples = data

    # --- before metrics ---
    before_poison = compute_poison_metrics_with_trigger(samples, trigger)
    before_clean = compute_clean_metrics(samples)

    # --- stage-2 restoration ---
    new_samples = []
    for s in tqdm(samples, desc=f"  {os.path.basename(input_path)}", leave=False):
        new_samples.append(process_sample(s, ppl_model, ppl_threshold))

    # --- after metrics ---
    after_poison = compute_poison_metrics_with_trigger(new_samples, trigger)
    after_clean = compute_clean_metrics(new_samples)

    # --- build output ---
    if isinstance(data, dict):
        output_data = dict(data)
        output_data["samples"] = new_samples
    else:
        output_data = {"samples": new_samples}

    output_data["stage2_config"] = {"ppl_threshold": ppl_threshold}
    output_data["metrics_before"] = {
        "poison": before_poison,
        "clean": before_clean,
    }
    output_data["metrics_after"] = {
        "poison": after_poison,
        "clean": after_clean,
    }
    output_data["localization_metrics"] = {
        "evaluated_count": after_poison["count"],
        "precision": after_poison["precision"],
        "recall": after_poison["recall"],
        "f1": after_poison["f1"],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    return before_poison, before_clean, after_poison, after_clean


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stage-2 PPL-based verification for deleted words"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Leaf directory to read from "
             "(e.g. /code/CBA-main/MyDetrctorLLM/results/translate/sleeper)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Leaf directory to write to "
             "(e.g. /code/CBA-main/MyDetrctorLLM/stage2_results)",
    )
    parser.add_argument(
        "--ppl_threshold",
        type=float,
        default=2.0,
        help="PPL difference threshold for keeping a deletion (default: 2.0)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="If specified, only process the file with this ratio "
             "(e.g. --ratio 0.50). Otherwise process all ratio files.",
    )
    parser.add_argument(
        "--trigger",
        type=str,
        required=True,
        help="Trigger text for computing P/R/F1. "
             "Comma-separated for distributed triggers (e.g. 'BadMagic,OpenAI,2024'), "
             "space-separated phrase otherwise (e.g. 'I watched a movie last night').",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if args.ratio is not None:
        target = os.path.join(
            input_dir,
            f"word_chunk_ablation_result_ratio_{args.ratio:.2f}.json",
        )
        if not os.path.isfile(target):
            print(f"[WARN] File not found: {target}")
            return
        file_list = [target]
    else:
        pattern = os.path.join(input_dir, "word_chunk_ablation_result_ratio_*.json")
        file_list = sorted(glob.glob(pattern))

    if not file_list:
        print(f"[WARN] No matching files in {input_dir}")
        return

    print(f"[INFO] Input dir:      {input_dir}")
    print(f"[INFO] Output dir:     {output_dir}")
    print(f"[INFO] PPL threshold:  {args.ppl_threshold}")
    print(f"[INFO] Trigger:        {args.trigger or '(use localization_metrics)'}")
    print(f"[INFO] Files found:    {len(file_list)}")

    print("[INFO] Loading GPT-2 ...")
    ppl_model = GPT2PPL()
    print("[INFO] GPT-2 loaded.\n")

    for fpath in file_list:
        fname = os.path.basename(fpath)
        out_path = os.path.join(output_dir, fname)

        bp, bc, ap, ac = process_file(fpath, out_path, ppl_model, args.ppl_threshold, args.trigger)

        print(f"{'=' * 70}")
        print(f"  {fname}")
        if bp["count"] > 0:
            print(f"  [Poison BEFORE] P={bp['precision']:.4f}  R={bp['recall']:.4f}  "
                  f"F1={bp['f1']:.4f}  ({bp['count']} samples)")
            print(f"  [Poison AFTER ] P={ap['precision']:.4f}  R={ap['recall']:.4f}  "
                  f"F1={ap['f1']:.4f}  ({ap['count']} samples)")
        if bc["count"] > 0:
            print(f"  [Clean  BEFORE] avg_removed={bc['avg_removed_words']:.2f}  "
                  f"avg_pct={bc['avg_removed_pct']:.2f}%  ({bc['count']} samples)")
            print(f"  [Clean  AFTER ] avg_removed={ac['avg_removed_words']:.2f}  "
                  f"avg_pct={ac['avg_removed_pct']:.2f}%  ({ac['count']} samples)")

    print(f"\n[DONE] Stage-2 results saved to: {output_dir}")


if __name__ == "__main__":
    main()
