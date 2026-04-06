#!/usr/bin/env python3
# dynamic_protocol_training.py
import os, json, glob, math, random, argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint

# LoRA (PEFT)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# Optional W&B
try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False


# =========================
# I/O helpers
# =========================

def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_protocols(p) -> List[str]:
    if p is None:
        return []
    if isinstance(p, list):
        return [str(x).strip() for x in p if str(x).strip()]
    s = str(p).strip()
    if not s:
        return []
    if "," in s:
        return [t.strip() for t in s.split(",") if t.strip()]
    return [s]

def dialog_to_text(dialog: List[Dict[str, Any]]) -> str:
    """Static (full) text: Role: utterance per line; drop protocol lines."""
    lines = []
    for turn in dialog or []:
        role = (turn.get("role") or "").strip()
        utt  = (turn.get("utterance") or "").strip()
        topic = (turn.get("topic") or "").strip()
        if topic and ("Exit to Protocol" in topic or "Protocol" in topic):
            continue
        if role or utt:
            lines.append(f"{role}: {utt}".strip(": "))
    return "\n".join(lines).strip()

def extract_lines_and_topics(dialog):
    """Return (lines, topics) parallel lists; drop explicit protocol lines from inputs."""
    lines, topics = [], []
    for turn in (dialog or []):
        role  = (turn.get("role") or "").strip()
        utt   = (turn.get("utterance") or "").strip()
        topic = (turn.get("topic") or "").strip()
        if topic and ("Exit to Protocol" in topic or "Protocol" in topic):
            # exclude from model input, but still used for divider detection (we use raw dialog for that)
            pass
        else:
            if role or utt:
                lines.append(f"{role}: {utt}".strip(": "))
                topics.append(topic)
    return lines, topics

def cumulative_join(lines, end_idx, start_idx=0):
    """Join lines[start_idx:end_idx] into a prefix string."""
    return "\n".join(lines[start_idx:end_idx])


# =========================
# Datasets
# =========================

class DiagDataset(Dataset):
    """Static: one full-text sample per dialogue JSON."""
    def __init__(self, files: List[str], tokenizer, label2id: Dict[str, int], max_length: int = 3072):
        self.files = files
        self.tok = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.num_labels = len(label2id)
        with open("/scratch/zar8jw/Conversation_Generation/data/realworld_ems/mapping.json", "r") as f:
            self.mapping = json.load(f)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = read_json(self.files[idx])
        text = dialog_to_text(obj.get("dialogue") or obj.get("dialog") or [])
        # labels = obj.get("protocol")
        enc = self.tok(
            text if text else "",
            truncation=True, max_length=self.max_length, padding=False, return_tensors=None,
        )
        # y = np.zeros(self.num_labels, dtype=np.float32)
        # if isinstance(labels, list):
        #     for lab in labels:
        #         if lab in self.mapping:
        #             lab = self.mapping[lab]
        #         if lab in self.label2id:
        #             y[self.label2id[lab]] = 1.0
        
        labs = obj.get("protocol")
        if labs is None:
            labs_list = []
        elif isinstance(labs, str):
            labs_list = [labs]
        elif isinstance(labs, list):
            labs_list = labs
        else:
            raise ValueError(f"Unexpected protocol type: {type(labs)} in {self.files[idx]}")
        
        y = np.zeros(self.num_labels, dtype=np.float32)
        for lab in labs_list:
            if lab in self.mapping:
                lab = self.mapping[lab]
            if lab in self.label2id:
                y[self.label2id[lab]] = 1.0

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": y,
            "length": len(enc["input_ids"]),
        }


class DynamicDiagDataset(Dataset):
    """
    Dynamic prefixes per dialogue.
      mode='last_k'        → use the last K prefixes (0-shot/cot/notechat).
      mode='post_protocol' → find first 'Exit to Protocol' in raw dialog; sample prefixes after it (+ full).
    """
    def __init__(
        self,
        files: List[str],
        tokenizer,
        label2id: Dict[str, int],
        mode: str = "last_k",
        last_k: int = 5,
        post_samples: int = 5,
        max_length: int = 3072,
        seed: int = 42,
    ):
        self.tok = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.mode = mode
        self.last_k = last_k
        self.post_samples = post_samples
        self.num_labels = len(label2id)
        self.rng = random.Random(seed)

        with open("/scratch/zar8jw/Conversation_Generation/data/realworld_ems/mapping.json", "r") as f:
            self.mapping = json.load(f)

        self.samples: List[Tuple[str, np.ndarray]] = []
        for path in files:
            obj = read_json(path)
            dialog = obj.get("dialogue") or obj.get("dialog") or []
            lines, _topics = extract_lines_and_topics(dialog)
            if not lines:
                continue

            # labels vector (common across prefixes)
            y = np.zeros(self.num_labels, dtype=np.float32)
            labs = obj.get("protocol")

            if labs is None:
                labs_list = []
            elif isinstance(labs, str):
                labs_list = [labs]
            elif isinstance(labs, list):
                labs_list = labs
            else:
                raise ValueError(f"Unexpected protocol type: {type(labs)} in {self.files[idx]}")

            for lab in labs_list:
                if lab in self.mapping:
                    lab = self.mapping[lab]
                if lab in label2id:
                    y[label2id[lab]] = 1.0

            N = len(lines)
            if N == 0:
                continue

            if self.mode == "last_k":
                start = max(1, N - self.last_k)  # at least one prefix
                for t in range(start, N):        # prefix lines[:t]
                    text = cumulative_join(lines, t)
                    if text.strip():
                        self.samples.append((text, y))

            elif self.mode == "post_protocol":
                # Find divider index in raw dialog (count non-protocol turns before it to map to filtered space)
                divider = None
                for i, tr in enumerate(dialog):
                    tp = (tr.get("topic") or "").strip()
                    if "Exit to Protocol" in tp:
                        divider = i
                        break

                if divider is not None:
                    nonprot_count = 0
                    for j in range(divider):
                        tp = (dialog[j].get("topic") or "").strip()
                        if tp and ("Exit to Protocol" in tp or "Protocol" in tp):
                            continue
                        nonprot_count += 1
                    candidates = list(range(max(1, nonprot_count + 1), N))
                    if candidates:
                        pick = candidates if len(candidates) <= self.post_samples \
                               else self.rng.sample(candidates, self.post_samples)
                        pick.sort()
                        for t in pick:
                            text = cumulative_join(lines, t)
                            if text.strip():
                                self.samples.append((text, y))
                # always include full prefix (up to last line)
                text_full = cumulative_join(lines, N)
                if text_full.strip():
                    self.samples.append((text_full, y))

            else:
                raise ValueError(f"Unknown dynamic mode: {self.mode}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        text, y = self.samples[idx]
        enc = self.tok(
            text, truncation=True, max_length=self.max_length, padding=False, return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": y.astype(np.float32),
            "length": len(enc["input_ids"]),
        }


# class EPCRDataset(Dataset):
#     """ePCR CSV to multi-label classification samples (unchanged)."""
#     def __init__(
#         self,
#         df: pd.DataFrame,
#         tokenizer,
#         label2id: Dict[str, int],
#         text_col: str = "Medic Note",
#         label_col: str = "Protocols",
#         max_length: int = 3072,
#     ):
#         self.df = df.reset_index(drop=True)
#         self.tok = tokenizer
#         self.label2id = label2id
#         self.text_col = text_col
#         self.label_col = label_col
#         self.max_length = max_length
#         self.num_labels = len(label2id)
#         with open("/scratch/zar8jw/Conversation_Generation/data/realworld_ems/mapping.json", "r") as f:
#             self.mapping = json.load(f)

#     def __len__(self): 
#         return len(self.df)

#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         row = self.df.iloc[idx]
#         text = str(row.get(self.text_col, "") or "").strip()
#         text = " ".join(text.split("//")[:3])  # keep first sections
#         raw = row.get(self.label_col, "")
#         labs = [p.strip() for p in str(raw).split(";") if p.strip()]
#         enc = self.tok(
#             text if text else "",
#             truncation=True, max_length=self.max_length, padding=False, return_tensors=None,
#         )
#         y = np.zeros(self.num_labels, dtype=np.float32)
#         for lab in labs:
#             if lab in self.mapping:
#                 lab = self.mapping[lab]
#             if lab in self.label2id:
#                 y[self.label2id[lab]] = 1.0
#         return {
#             "input_ids": enc["input_ids"],
#             "attention_mask": enc["attention_mask"],
#             "labels": y,
#             "length": len(enc["input_ids"]),
#         }

class DynamicEPCRDataset(Dataset):
    def __init__(self, df, tokenizer, label2id, last_k=5, max_length=3072, seed=42):
        self.tok = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.last_k = last_k
        self.num_labels = len(label2id)
        self.rng = random.Random(seed)

        with open("/scratch/zar8jw/Conversation_Generation/data/realworld_ems/mapping.json", "r") as f:
            self.mapping = json.load(f)

        self.samples = []  # list of (text, y)

        for _, row in df.iterrows():
            text = str(row.get("Medic Note", "") or "").strip()
            # e.g. split by '//' then sentences inside each block if you want
            chunks = [p.strip() for p in text.split("//") if p.strip()]
            if not chunks:
                continue

            # labels: same as EPCRDataset
            raw = row.get("Protocols", "")
            labs = [p.strip() for p in str(raw).split(";") if p.strip()]
            y = np.zeros(self.num_labels, dtype=np.float32)
            for lab in labs:
                if lab in self.mapping:
                    lab = self.mapping[lab]
                if lab in self.label2id:
                    y[self.label2id[lab]] = 1.0

            N = len(chunks)
            start = max(1, N - self.last_k)
            for t in range(start, N+1):
                prefix = "\n".join(chunks[:t])
                if prefix.strip():
                    self.samples.append((prefix, y.copy()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, y = self.samples[idx]
        enc = self.tok(
            text if text else "",
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": y.astype(np.float32),
            "length": len(enc["input_ids"]),
        }

# =========================
# Collator
# =========================

@dataclass
class PadCollator:
    tokenizer: Any
    def __call__(self, batch):
        max_len = max(x["length"] for x in batch)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            ids = x["input_ids"]; am = x["attention_mask"]
            pad_n = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad_n)
            attention_mask.append(am + [0] * pad_n)
            labels.append(x["labels"])
        labels_np = np.asarray(labels, dtype=np.float32)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.from_numpy(labels_np),
            "lengths": torch.tensor([x["length"] for x in batch], dtype=torch.long),
        }


# =========================
# Model (last-token head)
# =========================

class LastTokenClassifier(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.base = base_model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, lengths=None, **kwargs):
        out = self.base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h_last = out.hidden_states[-1]  # [B,T,H]
        idx = lengths - 1 if lengths is not None else attention_mask.sum(dim=1) - 1
        B, T, H = h_last.shape
        gather_index = idx.view(B, 1, 1).expand(B, 1, H)
        last_states = h_last.gather(dim=1, index=gather_index).squeeze(1)
        logits = self.classifier(self.dropout(last_states))
        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return {"loss": loss, "logits": logits}


# =========================
# Metrics
# =========================

def sigmoid(x): return 1 / (1 + np.exp(-x))

def multilabel_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5, ks=(1,3,5)) -> Dict[str, float]:
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(np.int32)
    tp = (preds & (labels == 1)).sum(); fp = (preds & (labels == 0)).sum(); fn = ((preds == 0) & (labels == 1)).sum()
    micro_p = tp / (tp + fp + 1e-9); micro_r = tp / (tp + fn + 1e-9)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)
    L = labels.shape[1]
    f1s = []
    for j in range(L):
        tp_j = (preds[:, j] & (labels[:, j] == 1)).sum()
        fp_j = (preds[:, j] & (labels[:, j] == 0)).sum()
        fn_j = ((preds[:, j] == 0) & (labels[:, j] == 1)).sum()
        p_j = tp_j / (tp_j + fp_j + 1e-9); r_j = tp_j / (tp_j + fn_j + 1e-9)
        f1s.append(2 * p_j * r_j / (p_j + r_j + 1e-9))
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    subset_acc = float((preds == labels).all(axis=1).mean())
    p_at, r_at = {}, {}
    for k in ks:
        topk_idx = np.argpartition(-probs, kth=min(k-1, probs.shape[1]-1), axis=1)[:, :k]
        precs, recs = [], []
        for i in range(labels.shape[0]):
            true_set = set(np.where(labels[i] == 1)[0].tolist())
            pred_set = set(topk_idx[i].tolist())
            inter = len(true_set & pred_set)
            precs.append(inter / k)
            recs.append(inter / (len(true_set) + 1e-9))
        p_at[k] = float(np.mean(precs)); r_at[k] = float(np.mean(recs))
    out = {"micro_f1": float(micro_f1), "macro_f1": float(macro_f1), "subset_acc": subset_acc}
    for k in ks:
        out[f"P@{k}"] = p_at[k]; out[f"R@{k}"] = r_at[k]
    return out


# =========================
# Trainer
# =========================

class CLSTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch: int = None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss


def make_compute_metrics(label2id: Dict[str, int]):
    def _compute(eval_pred):
        logits, labels = eval_pred
        logits = logits if isinstance(logits, np.ndarray) else logits[0]
        return multilabel_metrics(logits, labels, threshold=0.5, ks=(1,3,5))
    return _compute


# =========================
# Test (unchanged)
# =========================
# … keep your existing test() / test_turnwise() helpers here if you use them …
# (omitted for brevity—paste from your current script)

# --- helpers for turn-wise eval ---
def build_model_and_tok(args, num_labels: int):
    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # base LM
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype="auto", trust_remote_code=True, low_cpu_mem_usage=True
    )
    base.config.use_cache = False

    # Re-apply the SAME LoRA config as training
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj"], bias="none", task_type="CAUSAL_LM",
    )
    base = prepare_model_for_kbit_training(base)
    base = get_peft_model(base, lora_cfg)

    # Wrap with classifier head
    model = LastTokenClassifier(base, base.config.hidden_size, num_labels)
    return model, tok

def _load_state(model, checkpoint_dir: str):
    """
    Load weights from either:
      * Trainer checkpoint: pytorch_model.bin  (adapters+head state_dict)
      * Portable bundle: adapter_model.* + cls_head.pt
    """
    ckpt = Path(checkpoint_dir)

    # Portable bundle path
    adapter_ok = (ckpt / "adapter_config.json").exists()
    head_path = ckpt / "cls_head.pt"

    if adapter_ok and head_path.exists():
        # Attach adapters from checkpoint_dir
        # (we need the underlying base model component)
        model.base = PeftModel.from_pretrained(model.base, str(ckpt))
        # Load classifier head
        state = torch.load(head_path, map_location="cpu")
        model.classifier.load_state_dict(state)
        return

    # Fallback: Trainer checkpoint (single state_dict)
    bin_path = ckpt / "pytorch_model.bin"
    if not bin_path.exists():
        # support safetensors name if present
        st_path = ckpt / "model.safetensors"
        if st_path.exists():
            from safetensors.torch import load_file
            state = load_file(str(st_path))
        else:
            raise FileNotFoundError(f"No weights found in {checkpoint_dir}")
    else:
        state = torch.load(bin_path, map_location="cpu")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print(f"[warn] unexpected keys: {len(unexpected)} (ok for PEFT buffers)")
    if len(missing) > 0:
        print(f"[warn] missing keys: {len(missing)}")

def _lines_from_dialogue(obj):
    """Convert obj['dialogue' or 'dialog'] into ['Role: utterance', ...]."""
    dialog = obj.get("dialogue") or obj.get("dialog") or []
    out = []
    for turn in dialog:
        role = (turn.get("role") or "").strip()
        utt  = (turn.get("utterance") or "").strip()
        if role or utt:
            out.append(f"{role}: {utt}".strip(": "))
    return out

def _build_cumulative_contexts(lines, window=None):
    """Cumulative contexts per turn; if window is set, use sliding window."""
    ctxs = []
    for i in range(1, len(lines) + 1):
        start = 0 if window is None else max(0, i - window)
        ctxs.append("\n".join(lines[start:i]))
    return ctxs

@torch.no_grad()
def _predict_probs_batch(model, tok, device, texts, max_length=2048, batch_size=8):
    outs = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        enc = tok(
            chunk,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        lengths = attention_mask.sum(dim=1)
        out = model(input_ids=input_ids, attention_mask=attention_mask, lengths=lengths)
        logits = out["logits"] if isinstance(out, dict) else out.logits
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        outs.append(probs)
    return np.concatenate(outs, axis=0) if outs else np.zeros((0,))

def _labels_vector_from_obj(obj, label2id, mapping_path="/scratch/zar8jw/Conversation_Generation/data/realworld_ems/mapping.json"):
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    labs = obj.get("protocol")
    y = np.zeros(len(label2id), dtype=np.float32)
    if isinstance(labs, list):
        for lab in labs:
            lab = lab.lower().strip()
            if lab in mapping:
                lab = mapping[lab]
            if lab in label2id:
                y[label2id[lab]] = 1.0
    elif isinstance(labs, str):
        labs = labs.lower().strip()
        if labs in mapping:
            labs = mapping[labs]
        if labs in label2id:
            y[label2id[labs]] = 1.0
    else:
        raise Exception("check protocol:\n\n{obj}")
    return y



def test_turnwise(
    args,
    checkpoint_dir: str,
    test_glob_or_dir: str,
    out_npz: str,
    out_jsonl: str = None,
    threshold: float = 0.5,
    topk: int = 3,
    window: int = None,
):
    """
    Turn-by-turn (cumulative) evaluation. Saves NPZ with:
      files [N], turn_idx [N], logits [N,L], labels [N,L]
    Optionally a JSONL with top-k per turn for quick inspection.
    """
    # -------- label order (same as training) --------
    with open("/scratch/zar8jw/Conversation_Generation/data/realworld_ems/EMS_Protocol.json", "r") as f:
        label_list = json.load(f)
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}
    num_labels = len(label_list)
    if num_labels == 0:
        raise SystemExit("No labels discovered for testing.")

    # -------- collect test files --------
    spec = test_glob_or_dir
    if os.path.isdir(spec):
        spec = os.path.join(spec, "*.json")
    test_files = sorted(glob.glob(spec))
    if not test_files:
        raise SystemExit(f"No files matched: {test_glob_or_dir}")

    # -------- load model --------
    model, tok = build_model_and_tok(args, num_labels)
    _load_state(model, checkpoint_dir)
    print(f" ✅ successfully load model state")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    all_logits, all_labels, all_turn_idx, all_files = [], [], [], []
    jsonl_rows = []
    desc = f"TEST ({os.path.basename(args.test_checkpoint)})" if getattr(args, "test_checkpoint", None) else "test"
    for path in tqdm(test_files, desc=desc, total=len(test_files), dynamic_ncols=True):
        obj   = read_json(path)
        lines = _lines_from_dialogue(obj)
        if not lines:
            continue
        contexts = _build_cumulative_contexts(lines, window=window)   # T contexts
        probs = _predict_probs_batch(model, tok, device, contexts, max_length=args.max_length, batch_size=1)
        logits = np.log(probs + 1e-12) - np.log(1 - probs + 1e-12)    # invert sigmoid
        y_vec = _labels_vector_from_obj(obj, label2id)                 # [L]
        # print("probs: ", probs)
        # print("y_vec: ", y_vec)

        T = len(contexts)
        all_logits.append(logits)                                      # [T, L]
        all_labels.append(np.tile(y_vec, (T, 1)))                      # [T, L]
        all_turn_idx.append(np.arange(1, T+1, dtype=np.int32))         # [T]
        all_files.extend([path] * T)

        if out_jsonl:
            for t in range(T):
                row_probs = probs[t]
                idx = np.argpartition(-row_probs, kth=min(topk-1, len(row_probs)-1))[:topk]
                idx = idx[np.argsort(-row_probs[idx])]
                top = [(id2label[j], float(row_probs[j])) for j in idx]
                labs_above = [(id2label[j], float(row_probs[j])) for j in np.where(row_probs >= threshold)[0]]
                labs_above.sort(key=lambda x: -x[1])
                jsonl_rows.append({"file": path, "turn_index": int(t+1), "topk": top, "labels_above_thresh": labs_above})

    if not all_logits:
        raise SystemExit("No dialogs contained turns.")

    logits_cat = np.concatenate(all_logits, axis=0)   # [N,L]
    labels_cat = np.concatenate(all_labels, axis=0)   # [N,L]
    turns_cat  = np.concatenate(all_turn_idx, axis=0) # [N]
    files_arr  = np.array(all_files)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, files=files_arr, turn_idx=turns_cat, logits=logits_cat, labels=labels_cat)

    if out_jsonl:
        os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for r in jsonl_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[turnwise] saved NPZ -> {out_npz}" + (f" and JSONL -> {out_jsonl}" if out_jsonl else ""))
    return out_npz


def test(
    args,
    checkpoint_dir: str,
    test_glob_or_dir: str,
    out_dir: str,
    threshold: float = 0.5,
    topk: int = 3,
    window: int = None,
):
    """
    1) Runs turn-by-turn inference and saves NPZ (+ optional JSONL).
    2) Imports your evaluator and computes summary metrics.
    Returns: {"npz": <path>, "summary": <dict>, "per_dialog": <list>}
    """
    os.makedirs(out_dir, exist_ok=True)
    out_npz   = os.path.join(out_dir, "_turnwise.npz")
    out_jsonl = os.path.join(out_dir, "_turnwise.jsonl")
    # 1) save-only inference
    npz_path = test_turnwise(
        args=args,
        checkpoint_dir=checkpoint_dir,
        test_glob_or_dir=test_glob_or_dir,
        out_npz=out_npz,
        out_jsonl=out_jsonl,
        threshold=threshold,
        topk=topk,
        window=window,
    )

    # 2) compute metrics using your external evaluator
    from protocol_prediction_evaluate import evaluate_from_npz
    out = evaluate_from_npz(npz_path, prob_thresh=threshold)

    # pretty print (optional)
    print("\n=== Turnwise Summary ===")
    for k, v in out["summary"].items():
        print(f"{k}: {v}")


    metrics_json = os.path.join(out_dir, "_turnwise_metrics.json")
    per_dialog_jsonl = os.path.join(out_dir, "_turnwise_per_dialog.jsonl")
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump({"npz": npz_path, "summary": out["summary"]}, f, ensure_ascii=False, indent=4)
    with open(per_dialog_jsonl, "w", encoding="utf-8") as f:
        for row in out["per_dialog"]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


    return {"npz": npz_path, "summary": out["summary"], "per_dialog": out["per_dialog"]}



# =========================
# Train
# =========================

def safe_last_checkpoint(folder: str):
    if os.path.isdir(folder):
        try:
            if any(name.startswith("checkpoint-") for name in os.listdir(folder)):
                return get_last_checkpoint(folder)
        except Exception:
            pass
    return None

def train(args):
    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collate = PadCollator(tok)

    with open("/scratch/zar8jw/Conversation_Generation/data/realworld_ems/EMS_Protocol.json", "r") as f:
        label_list = json.load(f)
    label2id = {lab: i for i, lab in enumerate(label_list)}
    num_labels = len(label_list)

    if args.use_epcr:
        print("load ePCR for training")
        if not args.epcr_csv or not args.epcr_csv.strip():
            raise SystemExit("No CSV paths found. Provide --epcr_csv and set --use_epcr.")
        csv_paths = [p.strip() for p in args.epcr_csv.split(",") if p.strip()]
        df_all = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

        idx = list(range(len(df_all))); random.shuffle(idx)
        n_eval = max(1, int(len(idx) * args.eval_ratio))
        val_idx = set(idx[:n_eval])
        df_val   = df_all.iloc[[i for i in range(len(df_all)) if i in val_idx]].reset_index(drop=True)
        df_train = df_all.iloc[[i for i in range(len(df_all)) if i not in val_idx]].reset_index(drop=True)

        if args.dynamic:
            train_ds = DynamicEPCRDataset(df=df_train, tokenizer=tok, label2id=label2id, last_k=args.last_k, max_length=args.max_length, seed=args.seed)
            eval_ds  = DynamicEPCRDataset(df=df_val,   tokenizer=tok, label2id=label2id, last_k=args.last_k, max_length=args.max_length, seed=args.seed)
        else:
            train_ds = EPCRDataset(df=df_train, tokenizer=tok, label2id=label2id, max_length=args.max_length)
            eval_ds  = EPCRDataset(df=df_val,   tokenizer=tok, label2id=label2id, max_length=args.max_length)

    else:
        spec = args.data_glob
        if os.path.isdir(spec):
            spec = os.path.join(spec, "*.json")
        files = sorted(glob.glob(spec))
        if not files:
            raise SystemExit(f"No files matched: {args.data_glob}")

        random.shuffle(files)
        n_eval = max(1, int(len(files) * args.eval_ratio))
        eval_files  = files[:n_eval]
        train_files = files[n_eval:]

        if args.dynamic:
            # decide mode by path
            spec_lc = args.data_glob.lower()
            if any(k in spec_lc for k in ["ours", "ours-gemini"]):
                dyn_mode = "post_protocol"
            else:
                dyn_mode = "last_k"
            print(f"[dynamic] mode={dyn_mode}  last_k={args.last_k}  post_samples={args.post_samples}")

            train_ds = DynamicDiagDataset(
                train_files, tok, label2id,
                mode=dyn_mode, last_k=args.last_k, post_samples=args.post_samples,
                max_length=args.max_length, seed=args.seed,
            )
            eval_ds  = DynamicDiagDataset(
                eval_files, tok, label2id,
                mode=dyn_mode, last_k=args.last_k, post_samples=args.post_samples,
                max_length=args.max_length, seed=args.seed,
            )
        else:
            train_ds = DiagDataset(train_files, tok, label2id, max_length=args.max_length)
            eval_ds  = DiagDataset(eval_files,  tok, label2id, max_length=args.max_length)

    print(f"✅ Datasets ready — Train samples: {len(train_ds)} | Val samples: {len(eval_ds)}")

    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype="auto", trust_remote_code=True, low_cpu_mem_usage=True,
    )
    base.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj"], bias="none", task_type="CAUSAL_LM",
    )
    base = prepare_model_for_kbit_training(base)
    base = get_peft_model(base, lora_cfg)

    hidden_size = base.config.hidden_size
    model = LastTokenClassifier(base, hidden_size, num_labels)

    report_to = []
    if args.use_wandb and WANDB_OK and is_main_process(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0):
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        report_to = ["wandb"]

    last_ckpt = safe_last_checkpoint(args.output_dir)
    resume_arg = last_ckpt if last_ckpt is not None else None

    steps_per_epoch = max(1, math.ceil(len(train_ds) / (args.batch_size * args.grad_accum)))
    eval_steps = max(50, steps_per_epoch)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="micro_f1",
        greater_is_better=True,
        bf16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=args.num_workers,
        report_to=report_to,
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed,
        remove_unused_columns=False,
        save_safetensors=False,
    )

    trainer = CLSTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collate,
        processing_class=tok,
        compute_metrics=make_compute_metrics(label2id),
    )

    trainer.train(resume_from_checkpoint=resume_arg)

    if trainer.is_world_process_zero:
        metrics = trainer.evaluate()
        print("[final eval]", metrics)
        trainer.save_model(args.output_dir)
        with open(os.path.join(args.output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump({"label2id": label2id}, f, indent=2, ensure_ascii=False)
        if args.use_wandb and WANDB_OK:
            wandb.log(metrics); wandb.finish()


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_glob", type=str, default=None)
    parser.add_argument("--use_epcr", action="store_true",
                        help="If set, use ePCR CSV dataset via --epcr_csv instead of JSONs.")
    parser.add_argument("--epcr_csv", type=str,
                        default="/scratch/zar8jw/Conversation_Generation/data/RAA_processed_all.csv")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="./qwen_lora_protocol")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_ratio", type=float, default=0.3)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="conv-protocol-prediction-dynamic-train")
    parser.add_argument("--wandb_run", type=str, default="")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)

    # NEW: dynamic training switches
    parser.add_argument("--dynamic", action="store_true",
                        help="Enable dynamic prefix training.")
    parser.add_argument("--last_k", type=int, default=5,
                        help="K for last_k mode (0-shot/cot/notechat).")
    parser.add_argument("--post_samples", type=int, default=5,
                        help="Number of sampled prefixes after first Exit-to-Protocol (ours/ours-gemini).")

    # (Optional) Testing flags if you keep your test() logic
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--test_checkpoint", type=str, default=None)
    parser.add_argument("--test_glob", type=str, default="/scratch/zar8jw/Conversation_Generation/data/realworld_ems/dialog")
    parser.add_argument("--save_preds", type=str, default=None)

    args = parser.parse_args()

    if args.test_only:
        set_seed(args.seed)

        if not args.test_checkpoint:
            raise SystemExit("When using --test_only, you must provide --test_checkpoint (checkpoint dir).")
        if not args.test_glob:
            raise SystemExit("When using --test_only, you must provide --test_glob (dir or glob of test JSONs).")

        # Where to write NPZ / JSONL outputs
        out_dir = args.save_preds or os.path.join(args.test_checkpoint, "turnwise_eval")
        os.makedirs(out_dir, exist_ok=True)

        print("[TEST] checkpoint :", args.test_checkpoint)
        print("[TEST] test_glob  :", args.test_glob)
        print("[TEST] out_dir    :", out_dir)

        result = test(
            args=args,
            checkpoint_dir=args.test_checkpoint,
            test_glob_or_dir=args.test_glob,
            out_dir=out_dir,
            threshold=0.5,
            topk=3,
            window=None,
        )
    else:
        train(args)
