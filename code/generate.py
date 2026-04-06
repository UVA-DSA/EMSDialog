
import os, multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
import os, sys
import re
import json
# import torch
from tqdm import tqdm
import time
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import csv
from pathlib import Path
# import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Callable, Union
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from collections import defaultdict
import hashlib
from textwrap import dedent
from vllm import LLM, SamplingParams
# import transformers
from transformers import AutoTokenizer
# Set seed for reproducibility
# torch.manual_seed(42)


# model_name_or_path = "m42-health/Llama3-Med42-70B"
# model_name_or_path = "ProbeMedicalYonseiMAILab/medllama3-v20"
# model_name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct"
# model_name_or_path = "meta-llama/Meta-Llama-3.1-405B-Instruct-FP8"
# model_name_or_path = "meta-llama/Llama-3.3-70B-Instruct"
# model_name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"



def initialize_model():
    GEN_BACKEND = None  # "openai" | "gemini" | "vllm" | "hf"
    if any(tag in model_name_or_path for tag in ["o3-2025","o4-mini","gpt"]):
        import openai
        load_dotenv("./api_key/openai.env")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        GEN_BACKEND = "openai"
    elif "gemini" in model_name_or_path:
        import google.generativeai as genai
        from google.generativeai import types
        load_dotenv("./api_key/gemini.env")
        api_key = os.getenv("GEMINI_API_KEY")
        GEN_BACKEND = "gemini"
    # elif "Qwen3" in model_name_or_path:
    #     from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    #     model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype="bfloat16", device_map="auto")
    #     GEN_BACKEND = "hf"
    #     print(f"load {model_name_or_path} successfully, think={think}")
        # max_new_tokens = 32768
        # return tokenizer, model, GEN_BACKEND, None, max_new_tokens
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        tp = len(vis.split(",")) if vis else 1
        if "Qwen3" in model_name_or_path:
            model = LLM(
                model=model_name_or_path,       # e.g. "Qwen/Qwen3-4B"
                dtype="bfloat16",               # or "half"
                tensor_parallel_size=tp,         # >1 if using multiple GPUs
                trust_remote_code=True,         # Qwen needs this
                enforce_eager=True,
                gpu_memory_utilization=0.90,    # let vLLM use most of the GPU RAM
            )
            VLLM_DEFAULT_PARAMS = SamplingParams(
                temperature=1.0,
                max_tokens=32768,                 # budget for <think> + JSON; raise if needed
            )
            max_new_tokens = 32768

        elif "llama" in model_name_or_path or "OpenBioLLM" in model_name_or_path or "m42-health" in model_name_or_path:
            MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
            model = LLM(
                model=model_name_or_path,       # e.g. "Qwen/Qwen3-4B"
                dtype="bfloat16",               # or "half"
                tensor_parallel_size=tp,         # >1 if using multiple GPUs
                trust_remote_code=True,         # Qwen needs this
                gpu_memory_utilization=0.90,    # let vLLM use most of the GPU RAM
                # max_model_len=8192, # if using 2-a100
                max_model_len=MAX_MODEL_LEN, # if using 1-h200
            )
            VLLM_DEFAULT_PARAMS = SamplingParams(
                temperature=1.0,
                top_p=0.75,
                top_k=150,
                max_tokens=MAX_MODEL_LEN,
            )
            max_new_tokens = MAX_MODEL_LEN
        GEN_BACKEND = "vllm"
        return tokenizer, model, GEN_BACKEND, VLLM_DEFAULT_PARAMS, max_new_tokens


def apply_gemini(messages, client, model, temperature=0.3):
    start_time = time.time()
    response = client.models.generate_content(
        model=model, 
        contents=messages[0]["content"],
        config=types.GenerateContentConfig(
            temperature=temperature
        )
    )
    end_time = time.time()

    t_infer = end_time - start_time
    time.sleep(5)
    return response.text, t_infer

def apply_chatgpt(messages, model_name_or_path, temperature=0.3, max_tokens=8192):
    if "o4-mini" in model_name_or_path or "o3-2025" in model_name_or_path: 
        start_time = time.time()
        response = openai.chat.completions.create(
            model=model_name_or_path,
            messages=messages,
            # max_completion_tokens=max_tokens,
        )
        end_time = time.time()
    elif "gpt" in model_name_or_path:
        start_time = time.time()
        response = openai.chat.completions.create(
            model=model_name_or_path,
            messages=messages,
            temperature=temperature,
            # max_tokens=max_tokens,
            # top_p=top_p,
        )
        end_time = time.time()
    # return response.choices[0].message["content"]
    t_infer = end_time - start_time
    return response.choices[0].message.content, t_infer

def apply_qwen(messages, tokenizer, model):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    # conduct text completion
    
    start_time = time.time()
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    end_time = time.time()
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    t_infer = end_time - start_time
    return content, t_infer

def apply_qwen_batch(messages_list, tokenizer, model):
    """
    messages_list: list of chat messages (each is a list[{"role":..,"content":..}, ...])
    returns: (contents: List[str], t_infer: float)
      - contents[i] is the post-thinking text (same as `content` from your single-sample version)
      - t_infer is the wall-clock time for the whole batch (divide by len batch if you want per-sample)
    """
    # Ensure decoder-only–safe settings
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 1) Build chat texts with thinking enabled
    texts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think  # keep your global flag
        )
        for msgs in messages_list
    ]

    # 2) Tokenize as a batch (LEFT padding)
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        pad_to_multiple_of=8,   # optional small perf win
    ).to(model.device)

    # With LEFT padding, new tokens for every row start at this shared offset:
    base = model_inputs.input_ids.shape[1]

    # 3) Generate once for the whole batch
    start_time = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
    t_infer = time.time() - start_time

    # 4) Slice tails and split at </think> (token id 151668)
    THINK_CLOSE_ID = 151668
    contents = []
    for i in range(len(texts)):
        out_ids = generated_ids[i, base:].tolist()
        try:
            idx = len(out_ids) - out_ids[::-1].index(THINK_CLOSE_ID)
        except ValueError:
            idx = 0  # no </think> found
        # Return only the post-thinking content (same as your single-sample version)
        # thinking_content = tokenizer.decode(out_ids[:idx], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(out_ids[idx:], skip_special_tokens=True).strip("\n")
        contents.append(content)

    return contents, t_infer

def _build_prompt_from_messages(messages, tokenizer,enable_thinking=True):
    kw = dict(tokenize=False, add_generation_prompt=True)
    # Qwen’s template supports enable_thinking; others ignore extra kw
    if "qwen" in model_name_or_path.lower():
        kw["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(messages, **kw)

def apply_vllm(messages, tokenizer, model, sampling_params=None, max_new_tokens=None, enable_thinking=True):
    sp = sampling_params
    if max_new_tokens is not None:
        sp = SamplingParams(**{**sp.__dict__, "max_tokens": max_new_tokens})
    prompt = _build_prompt_from_messages(messages, tokenizer, enable_thinking=enable_thinking)
    t0 = time.time()
    outs = model.generate([prompt], sp, use_tqdm=False)
    dt = time.time() - t0
    return outs[0].outputs[0].text.strip(), dt

def apply_vllm_batch(messages_list, tokenizer, model, sampling_params=None, max_new_tokens=None, enable_thinking=True):
    sp = sampling_params
    if max_new_tokens is not None:
        sp = SamplingParams(**{**sp.__dict__, "max_tokens": max_new_tokens})
    prompts = [_build_prompt_from_messages(m, tokenizer, enable_thinking=enable_thinking) for m in messages_list]
    t0 = time.time()
    outs = model.generate(prompts, sp, use_tqdm=False)
    dt = time.time() - t0
    return [o.outputs[0].text.strip() for o in outs], dt

def extract_json(response, pattern = r'\[.*?\]'):
    matches = re.findall(pattern, response, re.DOTALL)

    if not matches:
        print("No JSON object found in the text.")
        print(response)        
        return "no json", None

    # json_data = matches[0] if len(matches) == 1 else matches[-1]
    json_data = matches[0]

    try:
        # Load the JSON data
        data = json.loads(json_data)
        return None, data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        # print(response)
        # print(json_data)
        return e, json_data

def call_llm_batch(epcr_list):
    """
    Batched path. Returns (jsonfiles_list, t_infer_total).
    conv_list: List[List[str]]  — each item is a history up to a turn
    """

    sys_prompt = open("prompt/sys.txt", "r").read()
    raw_prompt = open("prompt/user.txt", "r").read()

    # Build messages for each history
    messages_list = []
    for epcr in epcr_list:
        prompt = raw_prompt.format(epcr=epcr)

        messages_list.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ])


    pattern = r'\{.*?\}'
    json_list = []
    t_total = 0.0
    
    responses = []
    
    if GEN_BACKEND == "openai":
        for msgs in messages_list:
            resp, t = apply_chatgpt(msgs, temperature=0.3)
            responses.append(resp)
            t_total += (t or 0.0)
        
    elif GEN_BACKEND == "gemini":
        for msgs in messages_list:
            resp, t = apply_gemini(msgs, temperature=0.3)
            responses.append(resp)
            t_total += (t or 0.0)

    elif GEN_BACKEND == "hf":
        responses, t_total = apply_qwen_batch(messages_list)

    else:
        try:
            responses, t_total = apply_vllm_batch(messages_list, sampling_params=VLLM_DEFAULT_PARAMS, enable_thinking=think)
        except Exception as e:
            print(f"[WARN] Full batch failed: {e}. Retrying with batch_size=4...")
            torch.cuda.empty_cache()
            responses, t_total = [], 0.0
            batch_size = 4

            for i in range(0, len(messages_list), batch_size):
                batch = messages_list[i : i + batch_size]
                out, t = apply_vllm_batch(
                    batch,
                    sampling_params=VLLM_DEFAULT_PARAMS,
                    enable_thinking=think
                )
                responses.extend(out)
                t_total += (t or 0.0)

    for msgs, resp in zip(messages_list, responses):
        error, jf = extract_json(resp, pattern)
        if error or not jf:
            jf, _ = handleError(msgs, resp, pattern)
            if not jf:
                print("After handling error, there is still no json file.")
                return resp, None

        json_list.append(jf)

    return json_list, t_total


# --- helpers -----------------------------------------------------------------

def _val(x) -> str:
    """Return a clean string for a scalar; empty string for NaN/None."""
    if pd.isna(x) or x is None:
        return ""
    s = str(x).strip()
    # collapse internal whitespace a bit
    return " ".join(s.split())

def _as_lines(x: str) -> str:
    """Format semicolon-separated sequences (e.g., vitals/time-series) as bullet-like lines."""
    if not x:
        return ""
    parts = [p.strip() for p in str(x).split(";") if p.strip()]
    return "\n".join(f"- {p}" for p in parts)

def build_epcr_block(row: pd.Series) -> str:
    """Compose a normalized, multi-section ePCR block for prompting."""
    call_type = _val(row.get("Call Type"))
    chief = _val(row.get("Chief Complaint"))
    vital = _as_lines(_val(row.get("Vital")))
    pain = _val(row.get("Pain"))
    procedure = _as_lines(_val(row.get("Procedure")))
    meds_current = _val(row.get("Current Taken Medication"))
    history = _val(row.get("Medical/Surgical History"))
    allergies = _val(row.get("Allergies"))
    medic_note = _val(row.get("Medic Note"))
    narrative = _val(row.get("Narrative"))
    protocol = _val(row.get("Protocols"))
    timeseries = _as_lines(_val(row.get("Time-series")))
    age = _val(row.get("Age"))
    gender = _val(row.get("Gender"))

    epcr = [
        f"Call Type: {call_type}",
        # f"Chief Complaint: {chief}",
        "Vital:",
        vital if vital else "-",
        f"Pain: {pain or '-'}",
        "Procedure:",
        procedure if procedure else "-",
        f"Current Taken Medication: {meds_current or '-'}",
        f"Medical/Surgical History: {history or '-'}",
        f"Allergies: {allergies or '-'}",
        "Medic Note:",
        medic_note or "-",
        "Narrative:",
        narrative or "-",
        f"Protocol: {protocol or '-'}",
        "Time-series:",
        timeseries if timeseries else "-",
        f"Patient: {age, gender}"
    ]
    # strip trailing blank lines
    return "\n".join([ln for ln in epcr if ln is not None])



def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    return text.strip()

def _extract_top_level_block(s: str) -> str | None:
    """
    Return the first balanced top-level JSON array/object substring, or None.
    Supports nested [] and {} and strings with escapes.
    """
    s = s.lstrip()
    # If the whole thing is already pure JSON, great:
    try:
        json.loads(s)
        return s
    except Exception:
        pass

    # Otherwise, find the first '[' or '{' and scan until it balances.
    start = None
    for i, ch in enumerate(s):
        if ch in "[{":
            start = i
            opener, closer = ("[", "]") if ch == "[" else ("{", "}")
            break
    if start is None:
        return None

    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
    return None

def extract_json(response: str):
    """
    Robust JSON extractor:
      - strips ``` fences,
      - finds first balanced top-level array or object,
      - returns (err, data) where err is None on success.
    """
    text = _strip_code_fences(response)
    block = _extract_top_level_block(text)
    if block is None:
        return ValueError("No top-level JSON array/object found"), None
    try:
        return None, json.loads(block)
    except json.JSONDecodeError as e:
        return e, None


# ---------- unified tag extractor helpers (text-only; no JSON parsing) ----------
_TAG_CACHE = {}
def _between(text: str, tag: str) -> str:
    """Return inner text between <tag>...</tag> (case-insensitive); strips ```json fences if present."""
    key = tag.lower()
    if key not in _TAG_CACHE:
        _TAG_CACHE[key] = re.compile(
            rf"<\s*{tag}\s*>\s*(.*?)\s*<\s*/\s*{tag}\s*>",
            re.IGNORECASE | re.DOTALL,
        )
    m = _TAG_CACHE[key].search(text or "")
    if not m:
        return ""
    inner = m.group(1).strip()
    fm = re.match(r"\s*```(?:json)?\s*([\s\S]*?)\s*```\s*\Z", inner, re.IGNORECASE)
    return (fm.group(1).strip() if fm else inner)

def extract_for_shape(resp_text: str, expect_shape: str):
    """
    Unified extractor for all roles. Returns (ok, payload_tuple) matching call_llm's contract:
      - expect_shape == "plan"        -> (plan_text,)
      - expect_shape == "plan_crit"       -> ({"plan": plan_text, "critiques": crit_text},)
      - expect_shape == "dialogue"     -> ({"dialogue": dialogue_text},)
      - expect_shape == "dialogue_crit"-> ({"dialogue": revised_dialogue, "critiques": crit_text},)
    """
    plan_txt = _between(resp_text, "plan")
    crit_txt = (_between(resp_text, "critique")
                or _between(resp_text, "critiques")
                or _between(resp_text, "change_log"))
    dialogue_txt = _between(resp_text, "dialogue")
    approved_txt = _between(resp_text, "approved")


    if expect_shape == "plan":
        return (bool(plan_txt), (plan_txt.strip(),))

    if expect_shape == "plan_crit":
        approved_flag = None
        if approved_txt:
            at = approved_txt.strip().lower()
            approved_flag = (at in {"true", "1", "yes"})
        ok = (approved_flag is not None) or bool(crit_txt)
        payload = {"approved": bool(approved_flag), "critiques": (crit_txt or "").strip()}
        return ok, (payload,)

    if expect_shape == "concept":
        error, data = extract_json(resp_text)
        return (bool(data), (data,))

    if expect_shape == "dialogue":
        return (bool(dialogue_txt), ({"dialogue": (dialogue_txt or "").strip()},))

    if expect_shape == "dialogue_crit":
        ok = bool(crit_txt) or bool(dialogue_txt)
        payload = {"dialogue": (dialogue_txt or "").strip(),
                   "critiques": (crit_txt or "").strip()}
        return ok, (payload,)
    
    if expect_shape == "verifier":
        return (bool(resp_text), (resp_text,))

    raise ValueError("expect_shape must be one of: array|object|dialogue|dialogue_crit|verifier")


def build_topic_flow(gcs: int) -> str:
    conscious = gcs > 8
    if conscious:
        seq = (
            "Introduction → Chief Complaint → Responsiveness Exam → "
            "(History of Present Illness, Pain Assessment) → Primary Assessment → Secondary Assessment → "
            "Exit to Protocol → (Repeated: Reassessments → Take Interventions) → "
            "Transport → Handoff"
        )
    else:
        seq = (
            "Introduction → Chief Complaint → Responsiveness Exam → "
            "Primary Assessment → Secondary Assessment → (History of Present Illness, Pain Assessment) → "
            "Exit to Protocol → (Repeated: Reassessments → Take Interventions) → "
            "Transport → Handoff"
        )

    notes = (
        'Co-occur: "Take Vital Signs" and "Take Interventions" may occur during '
        "Primary/Secondary/History of Present Illness; repeat vitals after any intervention and later "
        "for trending."
    )

    return dedent(f"""\
    **Topic Flow**: {seq}.
    **Notes**: {notes}
    """).strip()


def extract_json_llm(response, max_regen=10, sampling=None, max_token=None, tokenizer=None, model=None, GEN_BACKEND=None):
    exception, data = extract_json(response)
    if not exception:
        return data
    else:
        for attempt in range(1, max_regen + 1):
            user_prompt = (
                "The previous response had a JSON parsing error. "
                "Please provide a valid JSON response. For example:\n"
                '[{"topic":"","micro_intent":"","evidence":["",""]}, ...]\n\n'
                "Ensure the JSON is properly formatted without any extra text.\n\n"
                f"Response: {response}"
            )
            regen_messages = [{"role": "user", "content": user_prompt}]

            def _run_backend(msgs):
                if GEN_BACKEND == "openai":
                    return apply_chatgpt(msgs, temperature=0.3)
                elif GEN_BACKEND == "gemini":
                    return apply_gemini(msgs, temperature=0.3)
                elif GEN_BACKEND == "hf":
                    return apply_qwen(msgs, tokenizer, model)
                else:
                    return apply_vllm(msgs, tokenizer, model, sampling_params=sampling, max_new_tokens=max_token, enable_thinking=think)

            response, _ = _run_backend(regen_messages)
            error, data = extract_json(response)
            if not error:
                return data
            else:
                print("JSON extraction failed after retries.")
                return []



# ---------- updated call_llm ---------------------------------------------------
def call_llm(epcr, gcs, concepts, tokenizer, model, GEN_BACKEND, prev_plan=None, prev_dialog=None, critiques=None, role=None, max_regen: int = 2, sampling=None, max_token=None):
    """
    Planner (role='planner'):
      returns (plan_text, t_infer) where plan_text is inner text of <plan>..</plan>.
    Criticizer (role='criticizer'):
      returns ({"plan": plan_text, "critiques": crit_text}, t_infer).
    Dialoguer (role='dialoguer'):
      returns ({"dialogue": dialogue_text}, t_infer).
    Dialogue Critic (role='dialogue_critic'):
      returns ({"dialogue": revised_dialogue_text, "critiques": crit_text}, t_infer).

    If required tags are missing, retry up to `max_regen` times with stricter instructions.
    Falls back to empty payloads if still missing.
    """
    topic_flow = build_topic_flow(gcs)
    # ---- choose prompts + expected shape ----
    if role == "planner":
        sys_prompt = open("prompt/plan/planner_sys.txt", "r", encoding="utf-8").read()
        raw_prompt = open("prompt/plan/planner_user.txt", "r", encoding="utf-8").read()
        sys_prompt = sys_prompt.format(topic_flow=topic_flow)
        user_prompt = raw_prompt.format(epcr=epcr, concept=";".join(concepts))
        expect_shape = "plan"

    elif role == "criticizer":
        sys_prompt = open("prompt/plan/criticizer_sys.txt", "r", encoding="utf-8").read()
        raw_prompt = open("prompt/plan/criticizer_user.txt", "r", encoding="utf-8").read()
        prev_blob = prev_plan if isinstance(prev_plan, str) else json.dumps(prev_plan, ensure_ascii=False)
        sys_prompt = sys_prompt.format(topic_flow=topic_flow)
        user_prompt = raw_prompt.format(epcr=epcr, concept=";".join(concepts), prev_plan=prev_blob)
        expect_shape = "plan_crit"

    elif role == "plan_reviser":
        sys_prompt = open("prompt/plan/reviser_sys.txt", "r", encoding="utf-8").read()
        raw_prompt = open("prompt/plan/reviser_user.txt", "r", encoding="utf-8").read()
        sys_prompt = sys_prompt.format(topic_flow=topic_flow)
        user_prompt = raw_prompt.format(epcr=epcr, prev_plan=(prev_plan or ""), critiques=(critiques or ""))
        expect_shape = "plan"
    
    elif role == 'plan_extractor':
        sys_prompt = open("prompt/ner/sys.txt", "r", encoding="utf-8").read()
        raw_prompt = open("prompt/ner/user.txt", "r", encoding="utf-8").read()
        user_prompt = raw_prompt.format(text=prev_plan)
        expect_shape = "concept"

    elif role == "dialoguer":
        sys_prompt = open("prompt/dialogue/dialoguer_sys.txt", "r", encoding="utf-8").read()
        raw_prompt = open("prompt/dialogue/dialoguer_user.txt", "r", encoding="utf-8").read()
        plan_blob = prev_plan if isinstance(prev_plan, str) else json.dumps(prev_plan, ensure_ascii=False)
        sys_prompt = sys_prompt.format(topic_flow=topic_flow)
        user_prompt = raw_prompt.format(epcr=epcr, plan=plan_blob)
        expect_shape = "dialogue"

    elif role == "dialogue_critic":
        sys_prompt = open("prompt/dialogue/criticizer_sys.txt", "r", encoding="utf-8").read()
        raw_prompt = open("prompt/dialogue/criticizer_user.txt", "r", encoding="utf-8").read()
        dlg_blob = prev_plan if isinstance(prev_plan, str) else json.dumps(prev_plan, ensure_ascii=False)
        sys_prompt = sys_prompt.format(topic_flow=topic_flow)
        user_prompt = raw_prompt.format(epcr=epcr, concept=";".join(concepts), dialogue=dlg_blob)
        expect_shape = "dialogue_crit"
    
    elif role == "dialogue_reviser":
        sys_prompt = open("prompt/dialogue/reviser_sys.txt", "r", encoding="utf-8").read()
        raw_prompt = open("prompt/dialogue/reviser_user.txt", "r", encoding="utf-8").read()
        user_prompt = raw_prompt.format(epcr=epcr, prev_dialogue=(prev_dialog or ""), critiques=(critiques or ""))
        expect_shape = "dialogue"
    
    elif role == 'dialogue_extractor':
        sys_prompt = open("prompt/ner/sys.txt", "r", encoding="utf-8").read()
        raw_prompt = open("prompt/ner/user.txt", "r", encoding="utf-8").read()
        user_prompt = raw_prompt.format(text=prev_dialog)
        expect_shape = "concept"

    elif role == 'refiner':
        sys_prompt = open("prompt/refine/refiner_sys.txt", "r", encoding="utf-8").read()
        raw_prompt = open("prompt/refine/refiner_user.txt", "r", encoding="utf-8").read()
        sys_prompt = sys_prompt.format(topic_flow=topic_flow)
        user_prompt = raw_prompt.format(epcr=epcr, concept=";".join(concepts), dialogue=prev_dialog)
        expect_shape = "dialogue"
    
    elif role == 'refiner_critic':
        sys_prompt = open("prompt/refine/criticizer_sys.txt", "r", encoding="utf-8").read()
        raw_prompt = open("prompt/refine/criticizer_user.txt", "r", encoding="utf-8").read()
        sys_prompt = sys_prompt.format(topic_flow=topic_flow)
        user_prompt = raw_prompt.format(epcr=epcr, concept=";".join(concepts), dialogue=prev_dialog)
        expect_shape = "dialogue_crit"
    
    elif role == 'refiner_reviser':
        sys_prompt = open("prompt/refine/reviser_sys.txt", "r", encoding="utf-8").read()
        raw_prompt = open("prompt/refine/reviser_user.txt", "r", encoding="utf-8").read()
        user_prompt = raw_prompt.format(epcr=epcr, prev_dialogue=(prev_dialog or ""), critiques=(critiques or ""))
        expect_shape = "dialogue"
    
    elif role == 'concept_verifier':
        sys_prompt = "You are a verifier. Answer ONLY with JSON true or false."
        user_prompt = (
            "Question: Based on the given text, is the concept mentioned (negation is ok, we just need to check if the concept exists)?\n\n"
            f"CONCEPT: {concepts}\n\n"
            "TEXT:\n"
            f"{epcr}\n\n"
            "Return ONLY 'true' or 'false'."
        )
        expect_shape = "verifier"

    else:
        raise ValueError("role must be 'planner'|'criticizer'|'plan_reviser'|'plan_extractor'|'dialoguer'|'dialogue_critic'|'dialogue_reviser'|'dialogue_extractor'|'refiner'")

    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}]

    # ---- backend router ----
    def _run_backend(msgs):
        if GEN_BACKEND == "openai":
            return apply_chatgpt(msgs, temperature=0.3)
        elif GEN_BACKEND == "gemini":
            return apply_gemini(msgs, temperature=0.3)
        elif GEN_BACKEND == "hf":
            return apply_qwen(msgs, tokenizer, model)
        else:
            return apply_vllm(msgs, tokenizer, model, sampling_params=sampling, max_new_tokens=max_token, enable_thinking=think)

    # ---- first try ----
    response, t_infer = _run_backend(messages)
    ok, payload = extract_for_shape(response, expect_shape)
    if ok:
        return (*payload, t_infer)

    # ---- regen with strict tag-only prefix ----
    for attempt in range(1, max_regen + 1):
        print(f"{role} rerun for {attempt} time")

        if expect_shape == "plan":
            prefix = (
                "Output ONLY the following tagged block (no prose, no extra text):\n\n"
                "<plan>\n"
                "[{\"topic\":\"\",\"micro_intent\":\"\",\"speaker\":\"\",\"evidence\":[\"\",\"\"]}]\n"
                "</plan>\n"
            )
        elif expect_shape == "plan_crit":
            prefix = (
                "Output ONLY the following tagged blocks (no prose, no extra text):\n\n"
                "<approved>true|false</approved>\n"
                "<critique>\n"
                "[\"<brief bullets of what you fixed and why>\"]\n"
                "</critique>\n\n"
            )
        elif expect_shape == "concept":
            prefix = (
                "Output ONLY a JSON array (no extra text) in the following format:\n\n"
                "[\"concept1\", \"concept2\", ...]\n\n"
            )
        elif expect_shape == "dialogue":
            prefix = (
                "Output ONLY one tagged block (no extra text). "
                "Inside <dialogue>…</dialogue>, write newline-delimited lines in this exact pattern: "
                "<turn>. <Topic>; <micro_intent>; <Role>: <utterance>\n\n"
                "<dialogue>\n"
                "1. Introduction; introduction; Medic: Hi, I'm Alex with the rescue squad. What made you call 911 today?\n"
                "2. Chief Complaint; identify_primary_complaint; Patient: Chest pain and shortness of breath for thirty minutes.\n"
                "</dialogue>\n"
            )

        elif expect_shape == "dialogue_crit":  # dialogue_crit
            prefix = (
                "Output ONLY the following tagged blocks(<approved>true|false</approved>, <critique>...</critique>)"
                "Do not include these delimiters inside any field values. No extra text, no code fences."
                "<approved>...</approved>"
                "<critique>"
                "1. ..."
                "2. ..."
                "3. ..."
                "..."
                "</critique>\n\n"
            )
        
        elif expect_shape == "verifier":
            prefix = (
                "Return ONLY 'true' or 'false'.\n\n"
            )
        else:
            raise ValueError("expect_shape must be one of: plan|plan_crit|dialogue|dialogue_crit|refiner|verifier")
        regen_messages = [{"role": "system", "content": sys_prompt},
                          {"role": "user", "content": prefix + "\n" + user_prompt}]

        response, t2 = _run_backend(regen_messages)
        if t2:
            t_infer = t2
        ok, payload = extract_for_shape(response, expect_shape)
        if ok:
            return (*payload, t_infer)

    # ---- fallbacks ----
    if expect_shape == "plan":
        print("Planner: missing <plan> after retries.")
        return "", t_infer
    if expect_shape == "plan_crit":
        print("Criticizer: missing <plan>/<critique> after retries.")
        return {"plan": "", "critiques": ""}, t_infer
    if expect_shape == "concept":
        print("Concept extractor: missing JSON array after retries.")
        return [], t_infer
    if expect_shape == "dialogue":
        print("Dialoguer: missing <dialogue> after retries.")
        return {"dialogue": ""}, t_infer
    if expect_shape == "dialogue_crit":
        print("Dialogue_critic: missing <critique>/<dialogue> after retries.")
        return {"dialogue": "", "critiques": ""}, t_infer
    if expect_shape == "verifier":
        print("Verifier: missing response after retries.")
        return "", t_infer


def _hash_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def _to_text(x) -> str:
    if x is None:
        return ""
    return x if isinstance(x, str) else str(x)



# ---- A tiny yes/no verifier using your LLM stack ----
def llm_yesno_contains(text: str, concept: str, tokenizer, model, GEN_BACKEND) -> bool:
    """
    Ask the LLM to strictly answer whether `concept` is explicitly present/supported in `text`.
    Returns True/False. Keep it deterministic (temperature=0).
    """

    fake_gcs = 15
    out, _ = call_llm(text, fake_gcs, concept, tokenizer, model, GEN_BACKEND, role="concept_verifier")

    s = out if isinstance(out, str) else _to_text(out)
    s = s.strip().lower()
    # tolerate code fences or words around
    s = s.replace("```json", "").replace("```", "").strip()
    if "true" in s and "false" not in s:
        return True
    if "false" in s and "true" not in s:
        return False
    # fallback: try to parse a bare JSON literal
    try:
        return bool(json.loads(s))
    except Exception:
        return False

def make_concept_critique(
    plan_concepts: List[str],
    gt_concepts: List[str],
    epcr_text: str,
    generated_text: str,
    tokenizer,
    model,
    GEN_BACKEND,
) -> Tuple[str, Dict]:
    """
    Compare two concept lists and return (critique_text, report_dict).
    - plan_concepts: list of strings asserted by the plan
    - gt_concepts:   list of strings from ground truth
    Optional: pass embed_fn to enable semantic cosine matching (>= sim_threshold).
    Always returns a <critique>...</critique> string. Shows up to 25 items per section.
    """

    WORD = re.compile(r"[A-Za-z0-9]+")
    def _norm(s: str) -> str:
        s = (s or "").strip().lower()
        s = " ".join(s.split())
        return " ".join(WORD.findall(s))

    # ---------- normalize ----------
    plan_raw = [p for p in (plan_concepts or []) if str(p).strip()]
    gt_raw   = [g for g in (gt_concepts   or []) if str(g).strip()]
    plan_norm = [_norm(p) for p in plan_raw]
    gt_norm   = [_norm(g) for g in gt_raw]
    plan_set  = set([x for x in plan_norm if x])
    gt_set    = set([x for x in gt_norm if x])

    # ---------- initial syntactic screening ----------
    # rule: substring match in normalized space (lenient and fast)
    def in_set_like(c: str, pool: set) -> bool:
        if not c: return False
        if c in pool: return True
        for q in pool:
            if c in q or q in c:
                return True
        return False

    # hallucinated = in plan but not in GT (syntactically)
    hallucinated = sorted({c for c in plan_set if not in_set_like(c, gt_set)})
    # missing = in GT but not in plan (syntactically)
    missing = sorted({c for c in gt_set if not in_set_like(c, plan_set)})

    # ---------- LLM rechecks (FOR LOOPS) ----------
    verified_supported_by_llm = []  # hallucinated -> actually found in ePCR
    removed_from_missing_by_llm = []  # missing -> actually present in plan/dialogue

    # Recheck hallucinated against ePCR text
    kept_hallucinated = []
    for c in hallucinated:
        exists = llm_yesno_contains(epcr_text, c, tokenizer=tokenizer, model=model, GEN_BACKEND=GEN_BACKEND)
        if exists:
            verified_supported_by_llm.append(c)
        else:
            kept_hallucinated.append(c)
    hallucinated = kept_hallucinated

    # Recheck missing against plan/dialogue text
    kept_missing = []
    for c in missing:
        exists = llm_yesno_contains(generated_text, c, tokenizer=tokenizer, model=model, GEN_BACKEND=GEN_BACKEND)
        if exists:
            removed_from_missing_by_llm.append(c)
        else:
            kept_missing.append(c)
    missing = kept_missing

    # ---------- format critique ----------
    def _block(title: str, items: List[str]) -> str:
        if not items: return ""
        return f"{title}\n" + "".join(f"  - {x}\n" for x in items)

    parts = []
    if missing:
        parts.append(_block("Missing concepts (present in ground truth but not covered):", missing))
    if hallucinated:
        parts.append(_block("Hallucinated concepts (asserted but not supported by GT/ePCR):", hallucinated))

    if missing or hallucinated:
        parts.append("Revision Hints:\n"
                     "  - Add steps/evidence for missing items.\n"
                     "  - Remove or justify unsupported items with citations to ePCR.")

    critique = "\n".join([p for p in parts if p]).rstrip()
    ok = not (missing or hallucinated)

    return ok, critique if critique else "No issues."



def criticize_topic_flow(jsondata, rules_path="topic_flow_rules.json",
                         allowed_starts=("Introduction", "Dispatch")):
    """
    Minimal checker with unknown-topic handling.

    - jsondata: ["Intro", ...] or [{"topic":"Intro", ...}, ...]
    - strict_unknown: if True, any topic not in rules -> fail
                      if False, we just warn but still check others
    Returns: (ok: bool, critique: str)
    """
    # load rules and topic list
    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    rules = {k.strip(): [v2.strip() for v2 in v] for k, v in rules.items()}
    topic_set = set(rules.keys())

    # normalize topics
    topics = []
    if isinstance(jsondata, list):
        for it in jsondata:
            if isinstance(it, str):
                topics.append(it.strip())
            elif isinstance(it, dict) and "topic" in it:
                topics.append(str(it["topic"]).strip())
    if not topics:
        return False, "No topics found."

    issues = []

    # unknown topics
    unknown_idxs = [(i, t) for i, t in enumerate(topics) if t not in topic_set]
    if unknown_idxs:
        pos = ", ".join([f"{t!r}@{i}" for i, t in unknown_idxs])
        issues.append(f"Unknown topic(s) not in topic list: {pos}.")

    # start check
    if topics[0] not in allowed_starts:
        issues.append(f"Start topic is '{topics[0]}' (allowed: {list(allowed_starts)}).")

    # transition checks (skip transitions involving unknowns if not strict)
    for i in range(len(topics) - 1):
        cur_t, nxt_t = topics[i], topics[i+1]
        if cur_t not in topic_set or nxt_t not in topic_set:
            continue  # already warned above
        allowed_next = rules.get(cur_t, [])
        if nxt_t not in allowed_next:
            preview = ", ".join(allowed_next[:6]) + ("..." if len(allowed_next) > 6 else "")
            issues.append(f"Illegal transition {i}->{i+1}: '{cur_t}' → '{nxt_t}'. Allowed next: [{preview}]")

    if issues:
        return False, "Topic flow issues:\n- " + "\n- ".join(issues)
    return True, "Topic flow satisfies the rules."




def generate_plan(
    epcr_text: str,
    gcs: int,
    concepts: list,
    out_dir: str,
    tokenizer,
    model,
    GEN_BACKEND,
    sampling,
    max_token,
    critique_passes: int = 20,
    sleep_between_calls: float = 0.0,
    max_regen: int = 2,
    enable_concept_check: bool = True,
    enable_topic_flow_check: bool = True,
) -> str:
    """
    Build a plan from EPCR with an initial planner pass followed by optional critic passes.
    Saves only .txt files into out_dir:
      - epcr.txt               (input copy)
      - plan_0.txt             (initial plan)
      - critique_{j}.txt       (each critique pass)
      - plan_{j+1}.txt         (each revised plan, if any)
      - final_plan.txt         (last plan)
    Returns final_plan_text.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) initial draft
    plan_raw, _t0 = call_llm(epcr_text, gcs, concepts, tokenizer, model, GEN_BACKEND,
                             role="planner", max_regen=max_regen, sampling=sampling, max_token=max_token)
    plan_text = _to_text(plan_raw).strip()
    if not plan_text: return ""
    with open(os.path.join(out_dir, "plan_0.txt"), "w", encoding="utf-8") as f:
        f.write(plan_text)
    prev_hash = _hash_text(plan_text)

    for cnt in range(critique_passes):
        # ---------- 0) Convert to Json ----------
        jsondata = extract_json_llm(plan_text, sampling=sampling, max_token=max_token, tokenizer=tokenizer, model=model, GEN_BACKEND=GEN_BACKEND)

        # ---------- 1) Concept extraction + fact critique ----------
        if jsondata:
            evidence_text = "\n".join(
                "\n".join(str(x) for x in item["evidence"]) if isinstance(item.get("evidence"), list)
                else str(item["evidence"])
                for item in jsondata
                if isinstance(item, dict) and "evidence" in item
            )
        else:
            evidence_text = plan_text

        # ✅ defaults so disabled checks don't crash / block stopping logic
        fact_ok, fact_critique = True, ""
        tf_ok, tf_critique = True, ""

        if enable_concept_check:
            plan_concept, _ = call_llm(plan_text, gcs, [], tokenizer, model, GEN_BACKEND, role="plan_extractor",
                                    prev_plan=evidence_text, max_regen=max_regen, sampling=sampling, max_token=max_token)
            fact_ok, fact_critique = make_concept_critique(plan_concept, concepts, epcr_text, evidence_text, tokenizer, model, GEN_BACKEND)
        
        # ---------- 2) Topic extraction + topic-flow critique ----------
        if enable_topic_flow_check:
            tf_ok, tf_critique = criticize_topic_flow(jsondata)

        # ---------- 3) Save critiques ----------
        if enable_concept_check:
            with open(os.path.join(out_dir, f"plan_critique_fact_{cnt}.txt"), "w", encoding="utf-8") as f:
                f.write(fact_critique or "(no issues)")
        if enable_topic_flow_check:
            with open(os.path.join(out_dir, f"plan_critique_tf_{cnt}.txt"), "w", encoding="utf-8") as f:
                f.write(tf_critique or "(no issues)")
        
        # ---------- 4) Exit if both pass ----------
        all_ok = fact_ok and tf_ok  # works because defaults are True when disabled
        if all_ok:
            print(f"[generate_plan] plan approved at pass {cnt}.")
            break

        # If neither produced actionable feedback, stop
        if (enable_concept_check and not fact_critique) and (enable_topic_flow_check and not tf_critique):
            print(f"[generate_plan] critics produced no actionable feedback at pass {cnt}; stopping.")
            break
        if (not enable_concept_check or not fact_critique) and (not enable_topic_flow_check or not tf_critique):
            # More permissive: stop if nothing actionable from whichever checks are enabled
            print(f"[generate_plan] no actionable feedback at pass {cnt}; stopping.")
            break
        
        # ---------- 5) Merge critiques and revise ----------
        merged_critique = "\n".join([s for s in [fact_critique if not fact_ok else "", tf_critique if not tf_ok else ""] if s]).strip()
        revised, _ = call_llm(
            epcr_text, gcs, concepts, tokenizer, model, GEN_BACKEND,
            prev_plan=plan_text, critiques=merged_critique, role="plan_reviser",
            max_regen=max_regen, sampling=sampling, max_token=max_token
        )
        new_plan_text = _to_text(revised).strip()
        if not new_plan_text:
            print(f"[generate_plan] plan_reviser returned empty plan at pass {cnt}; stopping.")
            break
        
        # Write and check for convergence
        with open(os.path.join(out_dir, f"plan_{cnt+1}.txt"), "w", encoding="utf-8") as f:
            f.write(new_plan_text)
        new_hash = _hash_text(new_plan_text)

        # if new_hash == prev_hash:
        #     print(f"[generate_plan] plan unchanged after revision at pass {cnt}; stopping.")
        #     plan_text = new_plan_text
        #     break

        # Prepare next iteration
        plan_text, prev_hash = new_plan_text, new_hash
    # ---------- final ----------
    with open(os.path.join(out_dir, "final_plan.txt"), "w", encoding="utf-8") as f:
        f.write(plan_text)
    return plan_text


    # # 2) critic-gated revisions
    # for j in range(max(1, critique_passes)):
    #     # a) critic judges ONLY
    #     crit, _ = call_llm(epcr_text, gcs, "", tokenizer, model, GEN_BACKEND,
    #                        prev_plan=plan_text, role="criticizer",
    #                        max_regen=max_regen, sampling=sampling, max_token=max_token)
    #     if isinstance(crit, dict):
    #         approved = bool(crit.get("approved"))
    #         crit_text = _to_text(crit.get("critiques")).strip()
    #     else:
    #         s = _to_text(crit)
    #         approved = (_between(s, "approved").strip().lower() in {"true","1","yes"})
    #         crit_text = (_between(s, "critique") or "").strip()

    #     with open(os.path.join(out_dir, f"plan_critique_{j}.txt"), "w", encoding="utf-8") as f:
    #         f.write(crit_text)

    #     if approved:
    #         print(f"[generate_plan] plan approved at pass {j}.")
    #         break
    #     if not crit_text:
    #         print(f"[generate_plan] critic produced no actionable critique at pass {j}.")
    #         break

    #     # b) reviser fixes the plan using the critique
    #     revised, _ = call_llm(epcr_text, gcs, crit_text, tokenizer, model, GEN_BACKEND,
    #                           prev_plan=plan_text, critiques=crit_text, role="plan_reviser",
    #                           max_regen=max_regen, sampling=sampling, max_token=max_token)
    #     new_plan_text = _to_text(revised).strip()
    #     if not new_plan_text:
    #         print(f"[generate_plan] planner_revise returned empty plan at pass {j}; stopping.")
    #         break

    #     with open(os.path.join(out_dir, f"plan_{j+1}.txt"), "w", encoding="utf-8") as f:
    #         f.write(new_plan_text)

    #     new_hash = _hash_text(new_plan_text)
    #     if new_hash == prev_hash:
    #         print(f"[generate_plan] plan unchanged after revision at pass {j}; stopping.")
    #         plan_text = new_plan_text
    #         break

    #     plan_text, prev_hash = new_plan_text, new_hash

    # # final
    # with open(os.path.join(out_dir, "final_plan.txt"), "w", encoding="utf-8") as f:
    #     f.write(plan_text)
    # return plan_text



def dialogue_text_to_json(text: str, rules_path="topic_flow_rules.json") -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Parse lines like '12. Topic; micro; Role: utterance' into JSON items.
    Returns (items, errors)
    items: [{"idx":1,"topic":"...","micro_intent":"...","role":"...","utterance":"..."}]
    """

    # load rules and topic list
    with open(rules_path, "r", encoding="utf-8") as f:
        rules = json.load(f)
    rules = {k.strip(): [v2.strip() for v2 in v] for k, v in rules.items()}
    topic_set = set(rules.keys())

    mapping = {
        "Reassessment": "Reassessments",
        "Reassess": "Reassessments",
        "Procedures": "Take Interventions",
        "Take Procedures": "Take Interventions",
        "Procedure": "Take Interventions",
        "Medicines": "Take Interventions",
        "Take Medicines": "Take Interventions",
        "Medication": "Take Interventions",
        "Take Medication": "Take Interventions",
        "Interventions": "Take Interventions",
        "Vitals": "Take Vital Signs",
        "Vital Signs": "Take Vital Signs",
        "Take Vitals": "Take Vital Signs",
        "Reassess Vitals": "Reassessments",
        "HPI": "History of Present Illness",
        "Sample": "History of Present Illness",
    }

    items, errors = [], []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # optional leading index "N. "
        if ". " in line and line.split(". ", 1)[0].isdigit():
            _, line = line.split(". ", 1)

        # expect "Topic; micro; Role: utterance"
        parts = [p.strip() for p in line.split(";", 2)]
        if len(parts) < 3 or ":" not in parts[2]:
            errors.append(f"Malformed line: {raw}")
            continue

        topic = parts[0]
        if topic in mapping:
            topic = mapping[topic]

        micro = parts[1]
        role, utt = [x.strip() for x in parts[2].split(":", 1)]
        items.append({
            "topic": topic,
            "micro_intent": micro,
            "role": role,
            "utterance": utt,
        })

    for i, it in enumerate(items, 1):
        it["idx"] = i
    return items, errors


def dialogue_json_to_text(items: List[Dict[str, str]]) -> str:
    lines = []
    for it in items:
        i = it.get("idx", len(lines)+1)
        lines.append(f"{i}. {it['topic']}; {it['micro_intent']}; {it['role']}: {it['utterance']}")
    return "\n".join(lines)


def topics_from_dialogue_json(items: List[Dict[str, str]]) -> List[str]:
    return [it["topic"].strip() for it in items if "topic" in it]

def generate_dialogue(
    epcr_text: str,
    gcs: int, 
    concepts: str,
    plan_text: str,
    out_dir: str,
    tokenizer, 
    model, 
    GEN_BACKEND, 
    sampling=None, 
    max_token=None,
    max_regen: int = 2,
    run_critic: bool = True,
    critic_passes: int = 2,
    sleep_between_calls: float = 0.0,
    enable_concept_check: bool = True,
    enable_topic_flow_check: bool = True,
    enable_style_check: bool = True,
) -> str:
    """
    Use role='dialoguer' to realize a single <dialogue>...</dialogue> block, then
    (optionally) run role='dialogue_critic' for N passes to produce critiques and an
    optional revised <dialogue>.

    Writes:
      - dialogue.txt                      (initial)
      - dialogue_critique.txt             (appended across passes, if run_critic)
      - dialogue_rev_pass{j}.txt          (when a revision is produced)
      - final_dialogue.txt                (revised if changed, else original)

    Returns the final dialogue text.
    """
    # --- 1) Generate initial dialogue ---
    out, _t = call_llm(epcr_text, gcs, concepts, tokenizer, model, GEN_BACKEND, prev_plan=plan_text, role="dialoguer", max_regen=max_regen, sampling=sampling, max_token=max_token)
    # Normalize to plain text
    if isinstance(out, dict):
        dialogue_txt = (out.get("dialogue") or "").strip()
    else:
        s = _to_text(out)
        dialogue_txt = (_between(s, "dialogue") or s).strip()

    # always parse text to JSON
    items, parse_errs = dialogue_text_to_json(dialogue_txt)
    cur_text = dialogue_json_to_text(items)   # stable pretty text

    with open(os.path.join(out_dir, "dialogue_0.txt"), "w", encoding="utf-8") as f:
        f.write(cur_text)


    prev_hash = _hash_text(cur_text)


    # If no critic desired or no dialogue produced, finalize and return
    if not run_critic or not items:
        with open(os.path.join(out_dir, "final_dialogue.txt"), "w", encoding="utf-8") as f:
            f.write(cur_text)
        return cur_text

    cur_items = items
    for j in range(max(1, critic_passes)):
        if sleep_between_calls:
            time.sleep(sleep_between_calls)

        # ✅ defaults so disabled checks won't crash / block
        fact_ok, fact_critique = True, ""
        tf_ok, tf_critique = True, ""
        style_ok, style_critique = True, ""

        # A) concept check
        if enable_concept_check:
            cur_content = "\n".join([each["utterance"] for each in cur_items])
            concept_extracted, _ = call_llm(cur_text, gcs, [], tokenizer, model, GEN_BACKEND, role="dialogue_extractor", 
                                        prev_dialog=cur_content, max_regen=max_regen, sampling=sampling, max_token=max_token)
            fact_ok, fact_critique = make_concept_critique(concept_extracted, concepts, epcr_text, cur_content, tokenizer, model, GEN_BACKEND)
            fact_critique = (fact_critique or "").strip()

        # B) topic flow check
        if enable_topic_flow_check:
            tf_ok, tf_critique = criticize_topic_flow(cur_items)
            tf_critique = (tf_critique or "").strip()

        # C) style check via LLM
        if enable_style_check:
            style_out, _ = call_llm(
                epcr_text, gcs, concepts, tokenizer, model, GEN_BACKEND,
                prev_dialog=cur_text, role="dialogue_critic",
                max_regen=max_regen, sampling=sampling, max_token=max_token
            )
            if isinstance(style_out, dict):
                style_critique = _to_text(style_out.get("critiques")).strip()
                style_ok = bool(style_out.get("approved"))
            else:
                s = _to_text(style_out)
                style_ok = (_between(s, "approved").strip().lower() in {"true","1","yes"})
                style_critique = (_between(s, "critique") or "").strip()

            # include parse errors as style feedback
            if parse_errs:
                parse_msg = "Formatting issues:\n- " + "\n- ".join(parse_errs)
                style_critique = (style_critique + ("\n" if style_critique else "") + parse_msg).strip()

        # save critiques
        if enable_concept_check:
            with open(os.path.join(out_dir, f"dialogue_critique_fact_{j}.txt"), "w", encoding="utf-8") as f:
                f.write(fact_critique or "(no issues)")
        if enable_topic_flow_check:
            with open(os.path.join(out_dir, f"dialogue_critique_topic_flow_{j}.txt"), "w", encoding="utf-8") as f:
                f.write(tf_critique or "(no issues)")
        if enable_style_check:
            with open(os.path.join(out_dir, f"dialogue_critique_style_{j}.txt"), "w", encoding="utf-8") as f:
                f.write(style_critique or "(no issues)")
        
        # exit if all pass
        all_ok = fact_ok and tf_ok  # (or: fact_ok and tf_ok and style_ok)
        if all_ok:
            print(f"[generate_dialogue] dialogue approved at pass {j}.")
            break


        # stop if nothing actionable from enabled checks + style
        actionable = []
        if enable_concept_check:
            actionable.append(fact_critique)
        if enable_topic_flow_check:
            actionable.append(tf_critique)
        actionable.append(style_critique)  # style always on here

        if not any(s.strip() for s in actionable):
            print(f"[generate_dialogue] critics produced no actionable feedback at pass {j}; stopping.")
            break
    
        # merge failing critiques
        merged_parts = []
        if enable_concept_check and (not fact_ok) and fact_critique:
            merged_parts.append(fact_critique)
        if enable_topic_flow_check and (not tf_ok) and tf_critique:
            merged_parts.append(tf_critique)
        merged = "\n".join(merged_parts).strip()


        # revise
        revised, _ = call_llm(
            epcr_text, gcs, concepts, tokenizer, model, GEN_BACKEND,
            prev_dialog=cur_text, critiques=merged, role="dialogue_reviser",
            max_regen=max_regen, sampling=sampling, max_token=max_token
        )
        
        if isinstance(revised, dict):
            new_text_raw = (revised.get("dialogue") or "").strip()
        else:
            s = _to_text(revised)
            new_text_raw = (_between(s, "dialogue") or s).strip()

        if not new_text_raw:
            print(f"[generate_dialogue] dialogue_reviser returned empty; stopping.")
            break
            
        # ALWAYS parse text to JSON (no auto-JSON extraction)
        new_items, parse_errs = dialogue_text_to_json(new_text_raw)
        new_text = dialogue_json_to_text(new_items)
        new_hash = _hash_text(new_text)

        with open(os.path.join(out_dir, f"dialogue_{j+1}.txt"), "w", encoding="utf-8") as f:
            f.write(new_text)

        # if new_hash == prev_hash:
        #     print(f"[generate_dialogue] dialogue unchanged after revision at pass {j}; stopping.")
        #     cur_items, cur_text, prev_hash = new_items, new_text, new_hash
        #     break

        cur_items, cur_text, prev_hash = new_items, new_text, new_hash

    # ---- final outputs ----
    with open(os.path.join(out_dir, "final_dialogue.txt"), "w", encoding="utf-8") as f:
        f.write(cur_text)


        


    # --- 2) Run dialogue_critic passes (optional) ---
    # cur = dialogue_txt
    # prev_hash = _hash_text(cur)

    # for j in range(max(1, critic_passes)):
    #     if sleep_between_calls:
    #         time.sleep(sleep_between_calls)

    #     crit_out, _tc = call_llm(epcr_text, gcs, concepts, tokenizer, model, GEN_BACKEND, prev_plan=cur, role="dialogue_critic", max_regen=max_regen, sampling=sampling, max_token=max_token)

    #     # Normalize critic output
    #     if isinstance(crit_out, dict):
    #         crit_text = _to_text(crit_out.get("critiques")).strip()
    #         approved = bool(crit_out.get("approved"))
    #     else:
    #         s = _to_text(crit_out)
    #         approved = (_between(s, "approved").strip().lower() in {"true","1","yes"})
    #         crit_text = (_between(s, "critique") or "").strip()

    #     # Save critique (append if multiple passes)
    #     if crit_text:
    #         with open(os.path.join(out_dir, f"dialogue_critique_{j}.txt"), 'w', encoding="utf-8") as f:
    #             f.write(crit_text)

    #     if approved:
    #         print(f"[generate_dialogue] dialogue approved at pass {j}.")
    #         break
    #     if not crit_text:
    #         print(f"[generate_dialogue] critic produced no actionable critique at pass {j}.")
    #         break

    #     # b) reviser fixes the dialogue using the critique
    #     revised, _ = call_llm(epcr_text, gcs, crit_text, tokenizer, model, GEN_BACKEND,
    #                           prev_dialog=dialogue_txt, critiques=crit_text, role="dialogue_reviser",
    #                           max_regen=max_regen, sampling=sampling, max_token=max_token)

    #     if isinstance(revised, dict):
    #         new_dialogue_text = (revised.get("dialogue") or "").strip()
    #     else:
    #         s = _to_text(revised)
    #         new_dialogue_text = (_between(s, "dialogue") or s).strip()

    #     if not new_dialogue_text:
    #         print(f"[generate_dialogue] dialogue_reviser returned empty dialogue at pass {j}; stopping.")
    #         break

    #     with open(os.path.join(out_dir, f"dialogue_{j+1}.txt"), "w", encoding="utf-8") as f:
    #         f.write(new_dialogue_text)

    #     new_hash = _hash_text(new_dialogue_text)
    #     if new_hash == prev_hash:
    #         print(f"[generate_dialogue] dialogue unchanged after revision at pass {j}; stopping.")
    #         dialogue_txt = new_dialogue_text
    #         break

    #     dialogue_txt, prev_hash = new_dialogue_text, new_hash

    # # final
    # with open(os.path.join(out_dir, "final_dialogue.txt"), "w", encoding="utf-8") as f:
    #     f.write(dialogue_txt)
    return cur_text


import os, time
from typing import List, Tuple

def refine_dialogue(
    epcr_text: str,
    gcs: int,
    concepts: List[str],           # GT concepts
    dialogue_text: str,
    tokenizer,
    model,
    GEN_BACKEND,
    n_pass: int,
    out_dir: str,
    max_regen: int = 5,
    sampling=None,
    max_token=None,
    sleep_between_calls: float = 0.0,
    enable_concept_check: bool = True,
    enable_topic_flow_check: bool = True,
    enable_style_check: bool = True,
) -> str:
    """
    Refine an existing dialogue using the same 3-check pipeline as generate_dialogue:
      1) concept check (LLM-aided mention detection)
      2) topic flow check
      3) style check via LLM

    Writes:
      - refine_0.txt, refine_1.txt, ...
      - refine_critique_fact_i.txt
      - refine_critique_topic_flow_i.txt
      - refine_critique_style_i.txt
      - final_dialogue_refined.txt
    """

    os.makedirs(out_dir, exist_ok=True)

    # --- normalize initial input ---
    dialogue_text = (dialogue_text or "").strip()
    items, parse_errs = dialogue_text_to_json(dialogue_text)
    cur_text = dialogue_json_to_text(items)
    prev_hash = _hash_text(cur_text)
    cur_items = items

    # if nothing parsed, just dump and exit
    if not cur_items:
        with open(os.path.join(out_dir, "final_dialogue_refined.txt"), "w", encoding="utf-8") as f:
            f.write(cur_text)
        return cur_text

    # save starting point
    with open(os.path.join(out_dir, "refine_0.txt"), "w", encoding="utf-8") as f:
        f.write(cur_text)

    for i in range(max(1, n_pass)):
        if sleep_between_calls:
            time.sleep(sleep_between_calls)

        fact_ok, fact_critique = True, ""
        tf_ok, tf_critique = True, ""
        style_ok, style_critique = True, ""

        # ===== A) concept check =====
        # current dialogue content (utterances only) for LLM extractor + for concept verifier
        cur_content = "\n".join(each.get("utterance", "") for each in cur_items)

        if enable_concept_check:
            # extract concepts from current dialogue
            extracted, _ = call_llm(
                cur_text, gcs, [], tokenizer, model, GEN_BACKEND,
                role="dialogue_extractor",
                prev_dialog=cur_content,
                max_regen=max_regen, sampling=sampling, max_token=max_token
            )
            # call your new concept critique (plan_concepts=extracted)
            fact_ok, fact_critique = make_concept_critique(
                plan_concepts=extracted,
                gt_concepts=concepts,
                epcr_text=epcr_text,
                generated_text=cur_content,
                tokenizer=tokenizer,
                model=model,
                GEN_BACKEND=GEN_BACKEND,
            )
            fact_critique = (fact_critique or "").strip()

        # ===== B) topic flow check =====
        if enable_topic_flow_check:
            tf_ok, tf_critique = criticize_topic_flow(cur_items)
            tf_critique = (tf_critique or "").strip()

        # ===== C) style check via LLM =====
        if enable_style_check:
            style_out, _ = call_llm(
                epcr_text, gcs, concepts, tokenizer, model, GEN_BACKEND,
                prev_dialog=cur_text, role="refiner_critic",
                max_regen=max_regen, sampling=sampling, max_token=max_token
            )
            if isinstance(style_out, dict):
                style_critique = _to_text(style_out.get("critiques")).strip()
                style_ok = bool(style_out.get("approved"))
            else:
                s = _to_text(style_out)
                style_ok = (_between(s, "approved").strip().lower() in {"true","1","yes"})
                style_critique = (_between(s, "critique") or "").strip()

            # add parse errors to style
            if parse_errs:
                parse_msg = "Formatting issues:\n- " + "\n".join(parse_errs)
                style_critique = (style_critique + ("\n" if style_critique else "") + parse_msg).strip()

        # ===== save critiques =====
        if enable_concept_check:
            with open(os.path.join(out_dir, f"refine_critique_fact_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(fact_critique or "(no issues)")
        if enable_topic_flow_check:
            with open(os.path.join(out_dir, f"refine_critique_topic_flow_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(tf_critique or "(no issues)")
        if enable_style_check:
            with open(os.path.join(out_dir, f"refine_critique_style_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(style_critique or "(no issues)")

        # ===== exit early if passes =====
        # (you only gated generate_dialogue on concept+topic; keep same rule here)
        all_ok = fact_ok and tf_ok and style_ok
        if all_ok:
            print(f"[refine_dialogue] dialogue approved at pass {i}.")
            break

        # If nobody produced actionable feedback, stop
        actionable = []
        if enable_concept_check:
            actionable.append(fact_critique)
        if enable_topic_flow_check:
            actionable.append(tf_critique)
        if enable_style_check:
            actionable.append(style_critique)
        if not any(s.strip() for s in actionable):
            print(f"[refine_dialogue] critics produced no actionable feedback at pass {i}; stopping.")
            break

        # ===== merge failing critiques =====
        merged_parts = []
        if enable_concept_check and (not fact_ok) and fact_critique:
            merged_parts.append(fact_critique)
        if enable_topic_flow_check and (not tf_ok) and tf_critique:
            merged_parts.append(tf_critique)
        if enable_style_check and (not style_ok) and style_critique:
            merged_parts.append(style_critique)  # include if you want style to drive revisions
        merged = "\n".join(merged_parts).strip()

        # ===== revise =====
        revised, _ = call_llm(
            epcr_text, gcs, concepts, tokenizer, model, GEN_BACKEND,
            prev_dialog=cur_text,
            critiques=merged,
            role="refiner_reviser",             # keep consistent with generate_dialogue
            max_regen=max_regen, sampling=sampling, max_token=max_token
        )

        if isinstance(revised, dict):
            new_text_raw = (revised.get("dialogue") or "").strip()
        else:
            s = _to_text(revised)
            new_text_raw = (_between(s, "dialogue") or s).strip()

        if not new_text_raw:
            print(f"[refine_dialogue] dialogue_reviser returned empty at pass {i}; stopping.")
            break

        # parse again
        new_items, parse_errs = dialogue_text_to_json(new_text_raw)
        new_text = dialogue_json_to_text(new_items)
        new_hash = _hash_text(new_text)

        # save intermediate
        with open(os.path.join(out_dir, f"refine_{i+1}.txt"), "w", encoding="utf-8") as f:
            f.write(new_text)

        # # optional: stop if unchanged
        # if new_hash == prev_hash:
        #     print(f"[refine_dialogue] dialogue unchanged after revision at pass {i}; stopping.")
        #     cur_items, cur_text, prev_hash = new_items, new_text, new_hash
        #     break

        # update loop vars
        cur_items, cur_text, prev_hash = new_items, new_text, new_hash

    # ---- final ----
    with open(os.path.join(out_dir, "final_dialogue_refined.txt"), "w", encoding="utf-8") as f:
        f.write(cur_text)
    return cur_text




def load_all_concepts(case_dir, case_id,
                      ner_root="/scratch/zar8jw/Conversation_Generation/log/NER/Qwen/Qwen3-32B/ePCR",
                      include_long_text=False):
    """
    ONE FUNCTION:
    - parse case_dir/epcr.txt (or other epcr files) → atomic concepts (vitals/procedures/etc.)
    - load NER concepts from {ner_root}/{case_id}.json
    - load local/log concepts from:
        case_dir/concepts.json
        case_dir/log/concepts.json
        case_dir/log/plan_concepts.json
        case_dir/log/dialogue_concepts.json
    - merge + dedup → return list[str]
    (time-series lines are IGNored)
    """
    import os, json

    def _clean(s):
        return " ".join((s or "").strip().split())

    def _load_json_list(path):
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return []
        if isinstance(data, list):
            return [_clean(x) for x in data if str(x).strip()]
        if isinstance(data, dict):
            return [_clean(x) for x in data.get("concepts", []) if str(x).strip()]
        return []

    # ----------------- vital / procedure helpers -----------------
    def _split_vital_timestamp(line):
        markers = ["pulse-", "resp-", "bp-", "glucose-", "spo2-", "ekg-"]
        idxs = []
        for m in markers:
            pos = line.find(":" + m)
            if pos != -1:
                idxs.append(pos)
        if not idxs:
            return None, line
        cut = min(idxs)
        ts = line[:cut]
        rest = line[cut + 1:]
        return ts, rest

    def _parse_vital_item(bullet_line):
        line = bullet_line.lstrip("-").strip()
        _ts, rest = _split_vital_timestamp(line)
        rest = rest.strip()
        if not rest:
            return []
        parts = rest.split()
        out = []
        i = 0
        while i < len(parts):
            w = parts[i]
            if w.startswith(("pulse-", "resp-", "bp-", "glucose-", "spo2-")):
                out.append(w)
                i += 1
                continue
            if w.startswith("ekg-"):
                tail = " ".join(parts[i:]).strip()
                if tail:
                    out.append(tail)
                break
            tail = " ".join(parts[i:]).strip()
            if tail:
                out.append(tail)
            break
        return out

    def _parse_procedure_item(bullet_line):
        line = bullet_line.lstrip("-").strip()
        parts = line.split(":", 3)
        if len(parts) == 4:
            return parts[3].strip()
        if ":" in line:
            return line.split(":", 1)[1].strip()
        return line

    # ----------------- EPCR parser -----------------
    def _parse_epcr_text(epcr_text):
        if not epcr_text:
            return []
        concepts = []
        cur_field = None

        for raw in epcr_text.splitlines():
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            stripped = line.strip()

            # bullets
            if stripped.startswith("- "):
                if cur_field and cur_field.lower().startswith("vital"):
                    concepts.extend(_parse_vital_item(stripped))
                elif cur_field and cur_field.lower().startswith("procedure"):
                    proc = _parse_procedure_item(stripped)
                    if proc and proc != "-":
                        concepts.append(proc)
                elif cur_field and cur_field.lower().startswith("time-series"):
                    # 👇 skip time-series bullets
                    continue
                else:
                    item = stripped[2:].strip()
                    if item:
                        concepts.append(item)
                continue

            # field line
            colon_pos = stripped.find(":")
            if colon_pos != -1 and colon_pos < 40:
                field = stripped[:colon_pos].strip()
                val = _clean(stripped[colon_pos + 1:])
                cur_field = field
                lf = field.lower()

                if lf.startswith("call type"):
                    if val and val != "-":
                        concepts.append("Call Type: " + val)
                elif lf.startswith("pain"):
                    if val and val != "-":
                        concepts.append("Pain: " + val)
                elif lf.startswith("medical/surgical history"):
                    if val and val != "-":
                        concepts.append("Medical/Surgical History: " + val)
                elif lf.startswith("allergies"):
                    if val and val != "-":
                        concepts.append("Allergies: " + val)
                elif lf.startswith("current taken medication") or lf.startswith("current medications"):
                    if val and val != "-":
                        concepts.append("Current Medications: " + val)
                elif lf.startswith("protocol"):
                    if val and val != "-":
                        concepts.append("Protocol: " + val)
                # Vital:, Procedure:, Medic Note:, Narrative:, Time-series: → handled below
                continue

            # continuation: Medic Note / Narrative (optional)
            if cur_field and (
                cur_field.lower().startswith("medic note")
                or cur_field.lower().startswith("narrative")
            ):
                if not include_long_text:
                    continue
                cont = _clean(stripped)
                if cont:
                    for chunk in cont.split("//"):
                        chunk = _clean(chunk)
                        if chunk:
                            concepts.append(cur_field + ": " + chunk)
                continue

            # continuation: Vital / Procedure
            if cur_field and cur_field.lower().startswith("vital"):
                concepts.extend(_parse_vital_item(stripped))
                continue
            if cur_field and cur_field.lower().startswith("procedure"):
                proc = _parse_procedure_item(stripped)
                if proc and proc != "-":
                    concepts.append(proc)
                continue

            # continuation: Time-series (non-bullet) → skip
            if cur_field and cur_field.lower().startswith("time-series"):
                continue

            # fallback
            stray = _clean(stripped)
            if stray:
                concepts.append(stray)

        # dedup
        out, seen = [], set()
        for c in concepts:
            c = _clean(c)
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        return out

    # ------------------------------------------------
    # merge sources
    # ------------------------------------------------
    all_concepts = []
    seen = set()

    # 1) NER concepts
    ner_path = os.path.join(ner_root, f"{case_id}.json")
    for c in _load_json_list(ner_path):
        if c and c not in seen:
            seen.add(c)
            all_concepts.append(c)

    # 2) epcr file
    epcr_text = ""
    with open(os.path.join(case_dir, case_id, "epcr.txt"), "r", encoding="utf-8") as f:
        epcr_text = f.read()

    if epcr_text:
        parsed = _parse_epcr_text(epcr_text)
        for c in parsed:
            if c and c not in seen:
                seen.add(c)
                all_concepts.append(c)

    # 3) local / log concepts
    log_candidates = [
        os.path.join(case_dir, "concepts.json"),
        os.path.join(case_dir, "log", "concepts.json"),
        os.path.join(case_dir, "log", "plan_concepts.json"),
        os.path.join(case_dir, "log", "dialogue_concepts.json"),
    ]
    for p in log_candidates:
        for c in _load_json_list(p):
            if c and c not in seen:
                seen.add(c)
                all_concepts.append(c)

    return all_concepts


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate Synthetic EMS Dialogues")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-32B", help="Name or path of the LLM (e.g. meta-llama/Llama-3.3-70B-Instruct or o4-mini-2025-04-16)")
    parser.add_argument("--start", type=int, default=0, help="Start index of the data to process")
    parser.add_argument("--end", type=int, default=-1, help="End index of the data to process (exclusive). Use -1 to indicate processing until the end")
    parser.add_argument("--enable_concept_check", action="store_true", help="if use concept check in Qwen3")
    parser.add_argument("--enable_topicflow_check", action="store_true", help="if use topic flow check in Qwen3")
    parser.add_argument("--enable_style_check", action="store_true", help="if use style check in Qwen3")
    parser.add_argument("--enable_think", action="store_true", help="if use think in Qwen3")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for inference")

    args = parser.parse_args()
    model_name_or_path = args.model_name_or_path
    start = args.start
    end = args.end
    think = args.enable_think
    enable_concept_check = args.enable_concept_check
    enable_topicflow_check = args.enable_topicflow_check
    enable_style_check = args.enable_style_check

    tokenizer, model, GEN_BACKEND, SAMPLING_PARAMS, max_new_tokens = initialize_model()

    csv_path = "/scratch/zar8jw/Conversation_Generation/data/RAA_processed_all.csv"
    max_rows = None  # set to e.g. 10 for quick testing
    df = pd.read_csv(csv_path)
    n = len(df)
    end = n if end == -1 else min(end, n)
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")

    subset = df.iloc[start:end]
    total = len(subset)

    log_dir = f"../log/{model_name_or_path}/ours"

    for i, row in tqdm(
        subset.iterrows(),
        desc=f"Plan → Generate → Refine [{start}:{end})",
        colour="blue",
        dynamic_ncols=True,
        total=total
    ):
        # case directory
        case_id = str(row.get("case_id", f"{i:06d}"))
        case_dir = os.path.join(log_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)

        if "final_dialogue_refined.txt" in os.listdir(case_dir):
            with open(os.path.join(case_dir, "final_dialogue_refined.txt"), "r", encoding="utf-8") as f:
                existing = f.read().strip()
            if existing:
                print(f"[{case_id}] already done; skipping.")
                continue

        # 0) read concepts and GCS score
        with open(os.path.join("/scratch/zar8jw/Conversation_Generation/log/gcs", f"{case_id}.gcs.json"), "r", encoding="utf-8") as f:
            gcs_dict = json.load(f)
        gcs = int(gcs_dict["total"]) if gcs_dict["total"] else 15

        # 1) Build EPCR text
        try:
            epcr_text = build_epcr_block(row)
        except Exception as e:
            raise Exception(f"[{case_id}] build_epcr_block failed: {e}")
        # keep a copy of inputs for reproducibility
        with open(os.path.join(case_dir, "epcr.txt"), "w", encoding="utf-8") as f:
            f.write(epcr_text)
        
        concepts = load_all_concepts(log_dir, case_id)
        # 2) Generate plan (and iterative plan critiques inside)

        if "final_plan.txt" not in os.listdir(case_dir):
            plan_text = generate_plan(
                epcr_text=epcr_text,
                gcs=gcs,
                concepts=concepts,
                out_dir=case_dir,
                tokenizer=tokenizer,
                model=model,
                GEN_BACKEND=GEN_BACKEND,
                sampling=SAMPLING_PARAMS,
                max_token=max_new_tokens,
                critique_passes=10,
                sleep_between_calls=0,
                max_regen=10,
                enable_concept_check=enable_concept_check,
                enable_topic_flow_check=enable_topicflow_check,
            )
        else:
            plan_text = open(os.path.join(case_dir, "final_plan.txt"), "r", encoding="utf-8").read().strip()

        if not plan_text:
            print(f"[{case_id}] empty plan; skipping dialogue.")
            raise Exception("empty plan")

        # 3) Generate dialogue (and optional dialogue_critic passes)

        if "final_dialogue.txt" in os.listdir(case_dir):
            dialogue_text = open(os.path.join(case_dir, "final_dialogue.txt"), "r", encoding="utf-8").read().strip()
        else:
            dialogue_text = generate_dialogue(
                epcr_text=epcr_text,
                gcs=gcs,
                concepts=concepts,
                plan_text=plan_text,
                out_dir=case_dir,
                tokenizer=tokenizer,
                model=model,
                GEN_BACKEND=GEN_BACKEND,
                sampling=SAMPLING_PARAMS,
                max_token=max_new_tokens,
                max_regen=10,
                run_critic=True,
                critic_passes=10,
                sleep_between_calls=0,
                enable_concept_check=enable_concept_check,
                enable_topic_flow_check=enable_topicflow_check,
            )

        if not dialogue_text:
            print(f"[{case_id}] empty dialogue produced.")
            raise Exception("empty dialogue")

        final_dialogue = refine_dialogue(
            epcr_text=epcr_text, 
            gcs=gcs,
            concepts=concepts,
            dialogue_text=dialogue_text, 
            tokenizer=tokenizer,
            model=model,
            GEN_BACKEND=GEN_BACKEND,
            n_pass=5,
            out_dir=case_dir,
            max_regen=10,
            sampling=SAMPLING_PARAMS,
            max_token=max_new_tokens,
            sleep_between_calls=0,
            enable_concept_check=enable_concept_check,
            enable_topic_flow_check=enable_topicflow_check,
            enable_style_check=enable_style_check,
        )

        if not final_dialogue:
            print(f"[{case_id}] empty refined dialogue produced.")
            raise Exception("empty refined dialogue")
        else:
            print(f"[{case_id}] ✅ plan + dialogue complete.")
