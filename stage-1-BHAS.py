
import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)


def load_model(
    base_model: str,
    quant: Optional[str] = None,
    device_map: str = "auto",
) -> Tuple[torch.nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        else:
            tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "torch_dtype": torch.float16,
    }

    if quant == "8bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quant not in (None, "none"):
        raise ValueError("--quant 只能取 none、8bit 或 4bit")

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    return model, tokenizer


def wrap_lora(base_model: torch.nn.Module, lora_weight: str) -> torch.nn.Module:
    print(f"[INFO] Loading LoRA from: {lora_weight}")
    model = PeftModel.from_pretrained(
        base_model,
        lora_weight,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model


def build_target_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, Any]:
    model, tokenizer = load_model(
        base_model=args.base_model,
        quant=args.quant,
        device_map=args.device_map,
    )

    if args.lora_path:
        model = wrap_lora(model, args.lora_path)


    model.eval()
    return model, tokenizer


def load_json_list(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError(f"文件 {path} 应当包含 JSON 列表。")

    return [item for item in data if isinstance(item, dict)]


def load_input_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not args.data_root:
        raise ValueError("必须提供 --data_root。")

    trigger_path = os.path.join(args.data_root, "trigger.json")
    no_trigger_path = os.path.join(args.data_root, "no_trigger.json")

    poison_examples = load_json_list(trigger_path) if os.path.isfile(trigger_path) else []
    clean_examples = load_json_list(no_trigger_path) if os.path.isfile(no_trigger_path) else []

    for ex in poison_examples:
        ex["is_poisoned"] = True
    for ex in clean_examples:
        ex["is_poisoned"] = False

    if args.num_samples is not None and args.num_samples>0:
        poison_examples = poison_examples[: args.num_samples]
        clean_examples = clean_examples[: args.num_samples]
    examples = poison_examples + clean_examples
    random.shuffle(examples)

    
    if not examples:
        raise ValueError(f"data_root={args.data_root} 下没有有效样本。")
    print(f"[INFO] 加载样本：poisoned={len(poison_examples)}, clean={len(clean_examples)}, 总计使用={len(examples)}")
    return examples


def get_input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def generate_responses(
    examples: Sequence[Dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    repetition_penalty: float,
    description: str = "生成回答",
) -> List[str]:
    prompts = [build_prompt_from_text(get_concatenated_user_content(ex)) for ex in examples]
    responses: List[str] = []
    input_device = get_input_device(model)

    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    for start in tqdm.tqdm(range(0, len(prompts), batch_size), desc=description):
        batch_prompts = prompts[start : start + batch_size]
        tokenized = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        input_ids = tokenized["input_ids"].to(input_device)
        attention_mask = tokenized["attention_mask"].to(input_device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )

        new_token_ids = output_ids[:, input_ids.shape[1] :]
        batch_responses = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        responses.extend(text.strip() for text in batch_responses)

    return responses


def encode_prompt_and_response(
    tokenizer: Any,
    prompt: str,
    response: str,
    append_eos: bool,
) -> Optional[Tuple[List[int], List[int], int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False, truncation=False)["input_ids"]

    if append_eos and tokenizer.eos_token_id is not None:
        if not response_ids or response_ids[-1] != tokenizer.eos_token_id:
            response_ids = response_ids + [tokenizer.eos_token_id]

    if not response_ids:
        return None

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    return input_ids, labels, len(response_ids)


def collate_scoring_batch(
    encoded_items: Sequence[Tuple[List[int], List[int], int]],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(item[0]) for item in encoded_items)
    batch_size = len(encoded_items)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)

    for row, (ids, item_labels, _) in enumerate(encoded_items):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(item_labels, dtype=torch.long, device=device)

    return input_ids, attention_mask, labels


def score_fixed_responses(
    prompts: Sequence[str],
    responses: Sequence[str],
    model: torch.nn.Module,
    tokenizer: Any,
    batch_size: int,
    append_eos: bool,
    description: str,
) -> List[Optional[Dict[str, float]]]:
    if len(prompts) != len(responses):
        raise ValueError("prompts 与 responses 的长度必须一致。")

    encoded: List[Optional[Tuple[List[int], List[int], int]]] = []

    valid_indices: List[int] = []

    for idx, (prompt, response) in enumerate(zip(prompts, responses)):
        item = encode_prompt_and_response(
            tokenizer=tokenizer,
            prompt=prompt,
            response=response,
            append_eos=append_eos,
        )

        encoded.append(item)

        if item is not None:
            valid_indices.append(idx)
        else:
            print('item is None')

    results: List[Optional[Dict[str, float]]] = [None] * len(prompts)

    device = get_input_device(model)

    for start in tqdm.tqdm(
        range(0, len(valid_indices), batch_size),
        desc=description,
    ):
        index_batch = valid_indices[start : start + batch_size]

        typed_batch = [
            encoded[idx]
            for idx in index_batch
            if encoded[idx] is not None
        ]

        input_ids, attention_mask, labels = collate_scoring_batch(
            encoded_items=typed_batch,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )

            logits = outputs.logits

        shift_logits = logits[:, :-1, :].float()

        shift_labels = labels[:, 1:]

        valid_mask = shift_labels.ne(-100)

        safe_labels = shift_labels.masked_fill(~valid_mask, 0)

        token_nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            safe_labels.reshape(-1),
            reduction="none",
        ).view_as(shift_labels)

        token_nll = token_nll * valid_mask

        total_nll = token_nll.sum(dim=1)

        token_count = valid_mask.sum(dim=1)

        mean_nll = total_nll / token_count.clamp_min(1)

        for local_row, original_idx in enumerate(index_batch):
            results[original_idx] = {
                "mean_nll": float(mean_nll[local_row].item()),

                "total_nll": float(total_nll[local_row].item()),

                "token_count": int(token_count[local_row].item()),
            }

        del outputs, logits, shift_logits, token_nll

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def split_text_to_words(text: str) -> List[str]:
    return str(text or "").strip().split()


def get_concatenated_user_content(example: Dict[str, Any]) -> str:
    instruction = str(example.get("instruction", "") or "").strip()
    input_text = str(example.get("input", "") or "").strip()
    if input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction


def _build_prompt_alpaca(user_content: str) -> str:
    instruction_part = user_content
    input_part = ""
    parts = user_content.split("\n\n", 1)
    if len(parts) == 2:
        instruction_part = parts[0]
        input_part = parts[1]

    if input_part:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction_part}\n\n"
            "### Input:\n"
            f"{input_part}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction_part}\n\n"
        "### Response:\n"
    )


def _build_prompt_llama2_chat(user_content: str) -> str:
    return (
        "<s>[INST] "
        "<<SYS>>\n"
        "You are a helpful, respectful and honest assistant.\n"
        "<</SYS>>\n\n"
        f"{user_content} "
        "[/INST] "
    )


_PROMPT_BUILDERS = {
    "alpaca": _build_prompt_alpaca,
    "llama2_chat": _build_prompt_llama2_chat,
}

_current_template = "alpaca"


def set_prompt_template(template_name: str):
    global _current_template
    if template_name not in _PROMPT_BUILDERS:
        raise ValueError(f"Unknown template: {template_name}, choose from {list(_PROMPT_BUILDERS.keys())}")
    _current_template = template_name


def build_prompt_from_text(user_content: str) -> str:
    return _PROMPT_BUILDERS[_current_template](user_content)


def build_ablated_prompt(
    words: List[str],
    spans: List[Tuple[int, int]],
) -> Tuple[str, str]:
    ablated_words: List[str] = []
    prev_end = 0
    for (s, e) in spans:
        ablated_words.extend(words[prev_end:s])
        prev_end = e
    ablated_words.extend(words[prev_end:])

    ablated_user_content = " ".join(ablated_words).strip()
    return build_prompt_from_text(ablated_user_content), ablated_user_content


def get_removed_text(words: List[str], spans: List[Tuple[int, int]]) -> str:
    parts = [" ".join(words[s:e]) for (s, e) in spans]
    return " | ".join(parts)


def generate_children_spans(
    spans: List[Tuple[int, int]],
    words: List[str],
) -> List[List[Tuple[int, int]]]:
    total = sum(e - s for s, e in spans)
    if total <= 1:
        return []

    A, B = _split_spans_at_midpoint(spans)
    if not A or not B:
        return []

    A_front, A_back = _split_spans_at_midpoint(A)
    B_front, B_back = _split_spans_at_midpoint(B)

    children: List[List[Tuple[int, int]]] = []
    seen: set = set()

    def _add(candidate: List[Tuple[int, int]]) -> None:
        c = sorted([(s, e) for s, e in candidate if e > s])
        if not c:
            return
        child_total = sum(e - s for s, e in c)
        if child_total >= total:
            return
        k = tuple(c)
        if k not in seen:
            seen.add(k)
            children.append(c)

    if B_front:
        _add(A + B_front)
    if A_back:
        _add(A_back + B)
    if B_back:
        _add(A + B_back)
    if A_front:
        _add(A_front + B)

    return children


def _split_spans_at_midpoint(
    spans: List[Tuple[int, int]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    total = sum(e - s for s, e in spans)
    half = total // 2
    if half <= 0:
        return [],[]

    front: List[Tuple[int, int]] = []
    back: List[Tuple[int, int]] = []
    accumulated = 0

    for (s, e) in spans:
        span_len = e - s
        if accumulated >= half:
            back.append((s, e))
        elif accumulated + span_len <= half:
            front.append((s, e))
            accumulated += span_len
        else:
            cut_point = s + (half - accumulated)
            front.append((s, cut_point))
            back.append((cut_point, e))
            accumulated = half

    return front, back


def recursive_search(
    words: List[str],
    spans: List[Tuple[int, int]],
    current_mnll: float,
    response: str,
    model: torch.nn.Module,
    tokenizer: Any,
    scoring_batch_size: int,
    append_eos: bool,
    binary_keep_ratio: float = 0.8,
    trace: Optional[List[Dict[str, Any]]] = None,
    depth: int = 0,
) -> Tuple[List[Tuple[int, int]], float, str, List[Dict[str, Any]]]:
    if trace is None:
        trace = []

    total_span_words = sum(e - s for s, e in spans)

    if total_span_words <= 0:
        trace.append({
            "depth": depth,
            "action": "stop_min_words",
            "spans": spans,
            "removed_text": get_removed_text(words, spans),
            "mnll": current_mnll,
            "total_words": total_span_words,
            "children": [],
        })
        return spans, current_mnll, get_removed_text(words, spans), trace

    children = generate_children_spans(spans, words)
    if not children:
        trace.append({
            "depth": depth,
            "action": "stop_no_children",
            "spans": spans,
            "removed_text": get_removed_text(words, spans),
            "mnll": current_mnll,
            "total_words": total_span_words,
            "children": [],
        })
        return spans, current_mnll, get_removed_text(words, spans), trace

    child_prompts = [build_ablated_prompt(words, c)[0] for c in children]
    child_responses = [response] * len(children)
    child_scores = score_fixed_responses(
        prompts=child_prompts,
        responses=child_responses,
        model=model,
        tokenizer=tokenizer,
        batch_size=scoring_batch_size,
        append_eos=append_eos,
        description="递归子节点评分",
    )

    children_info: List[Dict[str, Any]] = []
    best_idx = -1
    best_mnll = -float("inf")
    for ci, sc in enumerate(child_scores):
        child_mnll = float(sc["mean_nll"]) if sc is not None else None
        child_record = {
            "child_idx": ci,
            "spans": children[ci],
            "removed_text": get_removed_text(words, children[ci]),
            "total_words": sum(e - s for s, e in children[ci]),
            "mnll": child_mnll,
            "selected": False,
        }
        children_info.append(child_record)
        if child_mnll is not None and child_mnll > best_mnll:
            best_mnll = child_mnll
            best_idx = ci
    if best_idx < 0:
        trace.append({
            "depth": depth,
            "action": "stop_no_valid_child",
            "spans": spans,
            "removed_text": get_removed_text(words, spans),
            "mnll": current_mnll,
            "total_words": total_span_words,
            "children": children_info,
        })
        return spans, current_mnll, get_removed_text(words, spans), trace
    if current_mnll > 0 and best_mnll < binary_keep_ratio * current_mnll:
        children_info[best_idx]["selected"] = True
        trace.append({
            "depth": depth,
            "action": "stop_early_ratio",
            "spans": spans,
            "removed_text": get_removed_text(words, spans),
            "mnll": current_mnll,
            "total_words": total_span_words,
            "best_child_idx": best_idx,
            "best_child_mnll": best_mnll,
            "children": children_info,
        })
        return spans, current_mnll, get_removed_text(words, spans), trace

    children_info[best_idx]["selected"] = True
    trace.append({
        "depth": depth,
        "action": "recurse",
        "spans": spans,
        "removed_text": get_removed_text(words, spans),
        "mnll": current_mnll,
        "total_words": total_span_words,
        "best_child_idx": best_idx,
        "best_child_mnll": best_mnll,
        "children": children_info,
    })

    return recursive_search(
        words=words,
        spans=children[best_idx],
        current_mnll=best_mnll,
        response=response,
        model=model,
        tokenizer=tokenizer,
        scoring_batch_size=scoring_batch_size,
        append_eos=append_eos,
        binary_keep_ratio=binary_keep_ratio,
        trace=trace,
        depth=depth + 1,
    )


def run_detection(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    set_prompt_template(args.template)
    print(f"[INFO] Prompt template: {args.template}")

    examples = load_input_examples(args)
    print(f"[INFO] 待检测样本数量：{len(examples)}")

    model, tokenizer = build_target_model(args)

    prompts_all = [build_prompt_from_text(get_concatenated_user_content(ex)) for ex in examples]
    responses: List[str] = []
    input_device = get_input_device(model)
    generation_config = GenerationConfig(
        do_sample=False, num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    for start in tqdm.tqdm(range(0, len(prompts_all), args.generation_batch_size), desc="生成原始回答"):
        batch_prompts = prompts_all[start:start + args.generation_batch_size]
        tokenized = tokenizer(
            batch_prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=args.max_prompt_length,
        )
        input_ids = tokenized["input_ids"].to(input_device)
        attention_mask = tokenized["attention_mask"].to(input_device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                generation_config=generation_config,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
            )
        new_tokens = output_ids[:, input_ids.shape[1]:]
        responses.extend(t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True))

    original_scores = score_fixed_responses(
        prompts=prompts_all,
        responses=responses,
        model=model,
        tokenizer=tokenizer,
        batch_size=args.scoring_batch_size,
        append_eos=args.append_eos,
        description="计算原始 NLL",
    )

    sample_precomp: List[Optional[Dict[str, Any]]] = [None] * len(examples)
    
    for idx in tqdm.tqdm(range(len(examples)), desc="预计算每样本不变量"):
        example = examples[idx]
        response = responses[idx]
        orig_score = original_scores[idx]
        is_poisoned = bool(example.get("is_poisoned", False))

        if orig_score is None:
            continue

        original_nll = float(orig_score["mean_nll"])
        full_text = get_concatenated_user_content(example)
        words = split_text_to_words(full_text)
        num_words = len(words)

        if num_words < 2:
            continue

        root_spans = [(0, num_words)]
        root_prompt, _ = build_ablated_prompt(words, root_spans)
        root_score = score_fixed_responses(
            prompts=[root_prompt],
            responses=[response],
            model=model,
            tokenizer=tokenizer,
            batch_size=1,
            append_eos=args.append_eos,
            description="根节点评分",
        )[0]

        if root_score is None:
            continue

        sample_precomp[idx] = {
            "is_poisoned": is_poisoned,
            "original_nll": original_nll,
            "full_text": full_text,
            "words": words,
            "root_spans": root_spans,
            "root_mnll": float(root_score["mean_nll"]),
            "response": response
        }

    thresholds = [round(t * 0.05, 2) for t in range(1, 20)]

    trigger_parsed = None
    if args.trigger:
        if "," in args.trigger:
            parts = [t.strip() for t in args.trigger.split(",") if t.strip()]
            trigger_parsed = ("distributed", [p.lower().strip(".,!?;:'\"") for p in parts])
        else:
            trigger_parsed = ("phrase", [w.lower().strip(".,!?;:'\"") for w in args.trigger.split()])

    all_threshold_results: List[Dict[str, Any]] = []

    for thr in thresholds:
        print(f"\n{'='*60}")
        print(f"  BINARY_KEEP_RATIO = {thr:.2f}")
        print(f"{'='*60}")

        search_results: List[Optional[Dict[str, Any]]] = [None] * len(examples)

        for idx in tqdm.tqdm(range(len(examples)), desc=f"递归检测 (ratio={thr:.2f})"):
            precomp = sample_precomp[idx]
            if precomp is None:
                is_poisoned = bool(examples[idx].get("is_poisoned", False))
                search_results[idx] = {"valid": False, "is_poisoned": is_poisoned}
                continue

            final_spans, final_mnll, removed_text, search_trace = recursive_search(
                words=precomp["words"],
                spans=list(precomp["root_spans"]),
                current_mnll=0,
                response=precomp["response"],
                model=model,
                tokenizer=tokenizer,
                scoring_batch_size=args.scoring_batch_size,
                append_eos=args.append_eos,
                binary_keep_ratio=thr,
            )

            removed_text = get_removed_text(precomp["words"], final_spans)
            _, ablated_user_content = build_ablated_prompt(
                precomp["words"], final_spans,
            )

            search_results[idx] = {
                "valid": True,
                "is_poisoned": precomp["is_poisoned"],
                "original_nll": precomp["original_nll"],
                "final_spans": final_spans,
                "final_mnll": final_mnll,
                "removed_text": removed_text,
                "full_text": precomp["full_text"],
                "ablated_text_raw": ablated_user_content,
            }

        loc_metrics = None
        if trigger_parsed:
            trig_mode, trig_tokens = trigger_parsed
            tp_total, fp_total, fn_total = 0, 0, 0
            evaluated_count = 0

            for idx, r in enumerate(search_results):
                if not r or not r.get("valid") or not r.get("is_poisoned"):
                    continue

                full_text = r["full_text"]
                words_loc = split_text_to_words(full_text)
                num_words = len(words_loc)

                true_indices = set()
                words_lower = [w.lower().strip(".,!?;:'\"") for w in words_loc]

                if trig_mode == "distributed":
                    for t in trig_tokens:
                        for i, w in enumerate(words_lower):
                            if w == t:
                                true_indices.add(i)
                else:
                    tlen = len(trig_tokens)
                    for start in range(num_words - tlen + 1):
                        if all(words_lower[start + j] == trig_tokens[j] for j in range(tlen)):
                            for j in range(tlen):
                                true_indices.add(start + j)

                if not true_indices:
                    continue

                deleted_indices = set()
                for (s, e) in r["final_spans"]:
                    for j in range(s, e):
                        deleted_indices.add(j)

                punct_only = {i for i in range(num_words)
                              if all(c in ".,!?;:'\"()-[]{}/" for c in words_loc[i])}
                true_indices -= punct_only
                deleted_indices -= punct_only

                if not true_indices:
                    continue

                evaluated_count += 1
                tp_total += len(deleted_indices & true_indices)
                fp_total += len(deleted_indices - true_indices)
                fn_total += len(true_indices - deleted_indices)

            precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
            recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            loc_metrics = {
                "evaluated_count": evaluated_count,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            print(f"  [定位指标] ({evaluated_count} samples)  "
                  f"P={precision:.4f}  R={recall:.4f}  F1={f1:.4f}")

        per_sample: List[Dict[str, Any]] = []
        for idx in range(len(examples)):
            r = search_results[idx]
            if r is None or not r.get("valid"):
                is_p = r["is_poisoned"] if r else bool(examples[idx].get("is_poisoned", False))
                per_sample.append({"index": idx, "is_poisoned": is_p, "valid": False})
                continue
            per_sample.append({
                "index": idx,
                "is_poisoned": r["is_poisoned"],
                "valid": True,
                "detection_score": r["final_mnll"],
                "original_nll": r["original_nll"],
                "best_removed_text": r["removed_text"],
                "best_spans": r["final_spans"],
                "original_text": r["full_text"],
                "ablated_text_raw": r["ablated_text_raw"],
            })

        base_name, ext = os.path.splitext(args.output_file)
        thr_output_file = f"{base_name}_ratio_{thr:.2f}{ext}"
        thr_result = {
            "configuration": {
                "base_model": args.base_model,
                "lora_path": args.lora_path,
                "data_root": args.data_root,
                "binary_keep_ratio": thr,
                "seed": args.seed,
            },
            "localization_metrics": loc_metrics,
            "samples": per_sample,
        }
        output_dir = os.path.dirname(os.path.abspath(thr_output_file))
        os.makedirs(output_dir, exist_ok=True)
        with open(thr_output_file, "w", encoding="utf-8") as f:
            json.dump(thr_result, f, ensure_ascii=False, indent=2)
        print(f"  保存: {thr_output_file}")

        all_threshold_results.append({
            "binary_keep_ratio": thr,
            "localization_metrics": loc_metrics,
        })

    if args.trigger:
        print(f"\n{'='*70}")
        print(f"  多阈值定位指标汇总 (trigger='{args.trigger}')")
        print(f"{'='*70}")
        print(f"  {'Ratio':>8s}  {'Evaluated':>10s}  {'Precision':>10s}  {'Recall':>10s}  {'F1':>10s}")
        print(f"  {'-'*52}")
        for tr in all_threshold_results:
            m = tr["localization_metrics"]
            if m:
                print(f"  {tr['binary_keep_ratio']:8.2f}  {m['evaluated_count']:10d}"
                      f"  {m['precision']:10.4f}  {m['recall']:10.4f}  {m['f1']:10.4f}")

    summary_result = {
        "configuration": {
            "base_model": args.base_model,
            "lora_path": args.lora_path,
            "data_root": args.data_root,
            "thresholds": thresholds,
            "seed": args.seed,
        },
        "threshold_results": all_threshold_results,
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(summary_result, f, ensure_ascii=False, indent=2)

    print(f"\n========== negsenti 检测结果 ==========")
    print(f"样本: {len(examples)} | 阈值数: {len(thresholds)}")
    print(f"汇总: {args.output_file}")

    return summary_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="negsenti 词块删除 NLL 触发器检测器"
    )

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--quant", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--data_root", type=str, required=True,
                        help="数据目录，包含 trigger.json 和 no_trigger.json")
    parser.add_argument("--num_samples", type=int, default=None)

    parser.add_argument("--generation_batch_size", type=int, default=8)
    parser.add_argument("--scoring_batch_size", type=int, default=1)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--append_eos", action="store_true")

    parser.add_argument("--template", type=str, default="alpaca",
                        choices=["alpaca", "llama2_chat"],
                        help="Prompt 模板: alpaca 或 llama2_chat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--trigger", type=str, default=None,
                        help="已知触发器文本，用于计算词元级定位指标")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_detection(args)


if __name__ == "__main__":
    main()

