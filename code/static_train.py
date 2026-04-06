#!/usr/bin/env python3
import os, json, glob, math, random, argparse, re
from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from transformers.trainer_utils import is_main_process, get_last_checkpoint

from pathlib import Path

# LoRA (PEFT)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

# W&B (optional)
try:
    import wandb
    WANDB_OK = True
except Exception:
    WANDB_OK = False

# =========================
# Data loading & processing
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
    lines = []
    for turn in dialog or []:
        role = (turn.get("role") or "").strip()
        utt  = (turn.get("utterance") or "").strip()
        topic = (turn.get("topic") or "").strip()

        if topic:
            if topic == "Exit to Protocol" or "Protocol" in topic:
                continue
            
        if role or utt:
            lines.append(f"{role}: {utt}".strip(": "))
    return "\n".join(lines).strip()

# def collect_label_space(files: List[str]) -> List[str]:
#     labset = set()
#     for p in files:
#         obj = read_json(p)
#         labs = normalize_protocols(obj.get("protocol"))
#         labset.update(labs)
#     return sorted(labset)

class DiagDataset(Dataset):
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

        enc = self.tok(
            text if text else "",
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

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



class EPCRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        label2id: Dict[str, int],
        text_col: str = "Medic Note",
        label_col: str = "Protocols",
        max_length: int = 3072,
    ):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.label2id = label2id
        self.text_col = text_col
        self.label_col = label_col
        self.max_length = max_length
        self.num_labels = len(label2id)
        with open("/scratch/zar8jw/Conversation_Generation/data/realworld_ems/mapping.json", "r") as f:
            self.mapping = json.load(f)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        text = str(row.get(self.text_col, "") or "").strip()
        text = " ".join(text.split("//")[:3])

        # ePCR protocols are semicolon-separated in your CSV cells
        raw = row.get(self.label_col, "")
        labs = [p.strip() for p in str(raw).split(";") if p.strip()]

        enc = self.tok(
            text if text else "",
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        y = np.zeros(self.num_labels, dtype=np.float32)
        for lab in labs:
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




# @dataclass
# class PadCollator:
#     tokenizer: Any

#     def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
#         max_len = max(x["length"] for x in batch)
#         pad_id = self.tokenizer.pad_token_id

#         input_ids, attention_mask, labels = [], [], []
#         for x in batch:
#             ids = x["input_ids"]; am = x["attention_mask"]
#             pad_n = max_len - len(ids)
#             input_ids.append(ids + [pad_id] * pad_n)
#             attention_mask.append(am + [0] * pad_n)
#             labels.append(x["labels"])

#         return {
#             "input_ids": torch.tensor(input_ids, dtype=torch.long),
#             "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
#             "labels": torch.tensor(labels, dtype=torch.float32),
#             "lengths": torch.tensor([x["length"] for x in batch], dtype=torch.long),
#         }

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
            labels.append(x["labels"])  # numpy arrays

        labels_np = np.asarray(labels, dtype=np.float32)            # <- new
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.from_numpy(labels_np),                  # <- faster
            "lengths": torch.tensor([x["length"] for x in batch], dtype=torch.long),
        }


# =========================
# Model: last-token head
# =========================

class LastTokenClassifier(nn.Module):
    def __init__(self, base_model: AutoModelForCausalLM, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.base = base_model
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        lengths=None,
        **kwargs,
    ):
        out = self.base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        h_last = out.hidden_states[-1]  # [B,T,H]

        if lengths is not None:
            idx = lengths - 1
        else:
            idx = attention_mask.sum(dim=1) - 1

        B, T, H = h_last.shape
        gather_index = idx.view(B, 1, 1).expand(B, 1, H)
        last_states = h_last.gather(dim=1, index=gather_index).squeeze(1)  # [B,H]

        x = self.dropout(last_states)
        logits = self.classifier(x)  # [B,L]

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

    tp = (preds & (labels == 1)).sum()
    fp = (preds & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    micro_p = tp / (tp + fp + 1e-9)
    micro_r = tp / (tp + fn + 1e-9)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)

    L = labels.shape[1]
    f1s = []
    for j in range(L):
        tp_j = (preds[:, j] & (labels[:, j] == 1)).sum()
        fp_j = (preds[:, j] & (labels[:, j] == 0)).sum()
        fn_j = ((preds[:, j] == 0) & (labels[:, j] == 1)).sum()
        p_j = tp_j / (tp_j + fp_j + 1e-9)
        r_j = tp_j / (tp_j + fn_j + 1e-9)
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
# test
# =========================
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
        raise Exception(f"check protocol:\n\n{obj}")
    return y



def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x))

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
# Trainer
# =========================

class CLSTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int = None,
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            # e.g., transformers.ModelOutput
            logits = outputs.logits

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return (loss, outputs) if return_outputs else loss



def make_compute_metrics(label2id: Dict[str, int]):
    def _compute(eval_pred):
        logits, labels = eval_pred
        logits = logits if isinstance(logits, np.ndarray) else logits[0]
        return multilabel_metrics(logits, labels, threshold=0.5, ks=(1,3,5))
    return _compute

# =========================
# Train
# =========================


def safe_last_checkpoint(folder: str):
    # only try if directory exists AND appears to contain checkpoints
    if os.path.isdir(folder):
        try:
            # quick check to avoid listing errors or empty dirs
            if any(name.startswith("checkpoint-") for name in os.listdir(folder)):
                return get_last_checkpoint(folder)
        except Exception:
            pass
    return None

def train(args):
    set_seed(args.seed)
    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    collate = PadCollator(tok)

    with open("/scratch/zar8jw/Conversation_Generation/data/realworld_ems/EMS_Protocol.json", "r") as f:
        label_list = json.load(f)
    label2id = {lab: i for i, lab in enumerate(label_list)}
    num_labels = len(label_list)

    if args.use_epcr:
        # ePCR CSV mode
        print("load ePCR for training")
        if not args.epcr_csv or not args.epcr_csv.strip():
            raise SystemExit("No CSV paths found. Provide --epcr_csv and set --use_epcr.")
        csv_paths = [p.strip() for p in args.epcr_csv.split(",") if p.strip()]
        if not csv_paths:
            raise SystemExit("No CSV paths found in --epcr_csv")

        # Load all rows once; optional unlabeled drop
        df_all = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)

        # ---------- RANDOM SPLIT ONLY ----------
        idx = list(range(len(df_all)))
        random.shuffle(idx)
        n_eval = max(1, int(len(idx) * args.eval_ratio))
        val_idx = set(idx[:n_eval])
        df_val   = df_all.iloc[[i for i in range(len(df_all)) if i in val_idx]].reset_index(drop=True)
        df_train = df_all.iloc[[i for i in range(len(df_all)) if i not in val_idx]].reset_index(drop=True)

        # Datasets (split-first → then pass DF)
        train_ds = EPCRDataset(df=df_train, tokenizer=tok, label2id=label2id, max_length=args.max_length)
        eval_ds = EPCRDataset(df=df_val, tokenizer=tok, label2id=label2id, max_length=args.max_length)
    else:
        # files
        input_spec = args.data_glob
        print(f"Loading from {input_spec}")
        if os.path.isdir(input_spec):
            input_spec = os.path.join(input_spec, "*.json")
        files = sorted(glob.glob(input_spec))
        if not files:
            raise SystemExit(f"No files matched: {args.data_glob}")

        # split
        random.shuffle(files)
        n_eval = max(1, int(len(files) * args.eval_ratio))
        eval_files = files[:n_eval]
        train_files = files[n_eval:]

        # datasets
        train_ds = DiagDataset(train_files, tok, label2id, max_length=args.max_length)
        eval_ds  = DiagDataset(eval_files, tok, label2id, max_length=args.max_length)
    print(f"successfully load Train:{len(train_ds)} and Val:{len(eval_ds)}")

    # base LM
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base.gradient_checkpointing_enable()
    # # perf flags
    # base.config.use_cache = False  # needed for grad checkpointing
    # if args.gradient_checkpointing:
    #     try:
    #         base.gradient_checkpointing_enable()
    #     except Exception:
    #         pass
    if args.deepspeed:
        print(f"⚡ Using DeepSpeed config: {args.deepspeed}")
        base.config.use_cache = False

    # LoRA
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    base = prepare_model_for_kbit_training(base)
    base = get_peft_model(base, lora_cfg)

    hidden_size = base.config.hidden_size
    model = LastTokenClassifier(base, hidden_size, num_labels)

    # W&B (main process only)
    report_to = []
    if args.use_wandb and WANDB_OK and is_main_process(torch.distributed.get_rank() if torch.distributed.is_initialized() else 0):
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
        report_to = ["wandb"]

    last_ckpt = safe_last_checkpoint(args.output_dir)
    resume_arg = last_ckpt if last_ckpt is not None else None

    # training args — DDP/DeepSpeed handled by Trainer automatically when launched by torchrun
    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
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
        # fp16=args.fp16,
        bf16=True,  # if your GPUs support bf16
        dataloader_pin_memory=True,
        dataloader_num_workers=args.num_workers,
        report_to=report_to,
        ddp_find_unused_parameters=False,   # good default for LoRA + custom heads
        deepspeed=args.deepspeed,           # optional
        remove_unused_columns=False,        # we pass custom keys like 'lengths'
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

    # evaluate & save on main process only
    if trainer.is_world_process_zero:
        metrics = trainer.evaluate()
        print("[final eval]", metrics)
        trainer.save_model(args.output_dir)
        with open(os.path.join(args.output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump({"label2id": label2id}, f, indent=2, ensure_ascii=False)
        if args.use_wandb and WANDB_OK:
            wandb.log(metrics)
            wandb.finish()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_glob", type=str, default=None)
    parser.add_argument("--use_epcr", action="store_true", help="If set, use ePCR CSV dataset via --epcr_csv instead of JSONs.")
    parser.add_argument("--epcr_csv", type=str, default="/scratch/zar8jw/Conversation_Generation/data/RAA_processed_all.csv", help="Comma-separated CSV paths for ePCR dataset (if --use_epcr).")
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
    parser.add_argument("--wandb_project", type=str, default="conv-protocol-prediction")
    parser.add_argument("--wandb_run", type=str, default="")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed JSON config (optional)")
    parser.add_argument("--num_workers", type=int, default=4)
    # ✅ add this line so DDP/DeepSpeed can pass local rank
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by torch.distributed / deepspeed (ignored by this script).")

    parser.add_argument("--test_only", action="store_true", help="Run evaluation only on a chosen checkpoint")
    parser.add_argument("--test_checkpoint", type=str, default=None, help="Path to checkpoint folder, e.g. runs/.../checkpoint-200")
    parser.add_argument("--test_glob", type=str, default="/scratch/zar8jw/Conversation_Generation/data/realworld_ems/dialog", help='Glob/dir for test JSONs; if omitted, uses eval split')
    parser.add_argument("--save_preds", type=str, default=None, help="Path to save predictions (npz with logits & labels)")
    
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[warning] Ignoring unknown CLI arguments:", unknown)

    if args.test_only:
        if not args.test_checkpoint:
            raise SystemExit("--test_only requires --test_checkpoint")
        test_spec = args.test_glob if args.test_glob else (os.path.join(args.output_dir, "eval.json") if False else None)
        # Use your existing eval split if args.test_glob is None
        test_files = None
        if args.test_glob:
            spec = args.test_glob
            if os.path.isdir(spec): spec = os.path.join(spec, "*.json")
            test_files = sorted(glob.glob(spec))
            if not test_files:
                raise SystemExit(f"No files matched: {args.test_glob}")
        # run your helper
        test(
            args,
            checkpoint_dir=args.test_checkpoint,
            test_glob_or_dir=args.test_glob,
            out_dir=args.save_preds,
        )
    else:
        train(args)
