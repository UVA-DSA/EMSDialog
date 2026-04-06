#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# ---------------------- utils ----------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _first_commit_idx(pred_sets: np.ndarray) -> Optional[int]:
    """Index of first turn where any label is predicted (any 1)."""
    on = np.where(pred_sets.sum(axis=1) > 0)[0]
    return int(on[0]) if on.size else None

def _first_correct_idx(pred_sets: np.ndarray, y: np.ndarray) -> Optional[int]:
    """Index of first turn where prediction overlaps ground truth."""
    hit = np.where((pred_sets & y).sum(axis=1) > 0)[0]
    return int(hit[0]) if hit.size else None

def _count_set_changes(pred_sets: np.ndarray) -> int:
    """Number of set changes across turns (Hamming != 0 between consecutive sets)."""
    if len(pred_sets) <= 1:
        return 0
    diffs = (pred_sets[1:] ^ pred_sets[:-1]).any(axis=1)  # XOR then any change
    return int(diffs.sum())

def earliness_and_horizon(turns: np.ndarray,
                          first_idx: Optional[int],
                          first_correct_idx: Optional[int]) -> Dict[str, Optional[float]]:
    """
    commit_earliness   = 1 - t_commit / T
    commit_horizon     = T - t_commit
    correct_earliness  = 1 - t_correct / T
    correct_horizon    = T - t_correct
    """
    out = {
        "commit_earliness": None, "commit_horizon": None,
        "correct_earliness": None, "correct_horizon": None,
    }
    if turns.size == 0:
        return out
    T = int(turns[-1])
    if T <= 0:
        return out

    def comp(idx: Optional[int]) -> Tuple[Optional[float], Optional[int]]:
        if idx is None:
            return None, None
        t = int(turns[idx])
        return 1.0 - (t / float(T)), T - t

    out["commit_earliness"],  out["commit_horizon"]  = comp(first_idx)
    out["correct_earliness"], out["correct_horizon"] = comp(first_correct_idx)
    return out

def edit_overhead_sets(pred_sets: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], int, int, bool]:
    """
    Edit Overhead (EO) over the committed subsequence (post first commit) for multi-label sets.

    total_changes: number of times the predicted set changes after first commit.
    necessary: 1 iff the first committed set does NOT overlap GT and the sequence EVER reaches overlap later; else 0.
    Cases:
      - No commit -> (None, 0, 0, False)
      - total_changes == 0 -> EO = 0.0 if initial set overlaps GT else 1.0
      - Never overlaps GT -> EO = 1.0
      - Else -> EO = (total_changes - necessary) / total_changes
    """
    i0 = _first_commit_idx(pred_sets)
    if i0 is None:
        return None, 0, 0, False

    sub = pred_sets[i0:]                       # [S, L]
    total_changes = _count_set_changes(sub)

    init_overlap = bool((sub[0] & y).any())
    ever_overlap = bool((sub & y).any())

    if total_changes == 0:
        eo = 0.0 if init_overlap else 1.0
        return eo, total_changes, 0, True

    if not ever_overlap:
        return 1.0, total_changes, 0, True

    # Classical "one improvement" view: only the first arrival to overlapping GT can be necessary.
    necessary = 0 if init_overlap else 1
    eo = (total_changes - necessary) / float(total_changes)
    return eo, total_changes, necessary, True

def multilabel_prf1(preds_bin: np.ndarray, labels_bin: np.ndarray):
    """
    preds_bin, labels_bin: [N, L] 0/1 arrays
    Returns dict: micro_p, micro_r, micro_f1, macro_f1
    """
    preds_bin = preds_bin.astype(np.int32)
    labels_bin = labels_bin.astype(np.int32)
    # micro
    tp = int(((preds_bin == 1) & (labels_bin == 1)).sum())
    fp = int(((preds_bin == 1) & (labels_bin == 0)).sum())
    fn = int(((preds_bin == 0) & (labels_bin == 1)).sum())
    micro_p = tp / (tp + fp + 1e-9)
    micro_r = tp / (tp + fn + 1e-9)
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)

    # macro-F1 (per label F1, average)
    L = labels_bin.shape[1]
    f1s = []
    for j in range(L):
        tp_j = int(((preds_bin[:, j] == 1) & (labels_bin[:, j] == 1)).sum())
        fp_j = int(((preds_bin[:, j] == 1) & (labels_bin[:, j] == 0)).sum())
        fn_j = int(((preds_bin[:, j] == 0) & (labels_bin[:, j] == 1)).sum())
        p_j = tp_j / (tp_j + fp_j + 1e-9)
        r_j = tp_j / (tp_j + fn_j + 1e-9)
        f1_j = 2 * p_j * r_j / (p_j + r_j + 1e-9)
        f1s.append(f1_j)
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        "micro_p": float(micro_p),
        "micro_r": float(micro_r),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
    }

def compute_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """
    Standard ECE with uniform bins on [0,1].
    conf: [N] confidence values in [0,1]
    correct: [N] boolean correctness (True/False)
    """
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if conf.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # put conf==1.0 into last bin
    bin_ids = np.digitize(conf, edges, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    N = conf.size
    for b in range(n_bins):
        m = (bin_ids == b)
        nb = int(m.sum())
        if nb == 0:
            continue
        acc_b = float(correct[m].mean())
        conf_b = float(conf[m].mean())
        ece += abs(acc_b - conf_b) * (nb / N)
    return float(ece)


# ---------------------- per-dialog evaluation ----------------------

def eval_dialog_from_npz_block(turns: np.ndarray,
                               logits: np.ndarray,
                               y_vec: np.ndarray,
                               prob_thresh: float) -> Dict[str, Any]:
    """
    turns: [T]  (1-based)
    logits: [T, L]
    y_vec: [L]  (multi-hot GT)
    """
    T = turns.size
    probs = sigmoid(logits)                     # [T, L]
    pred_sets = (probs >= prob_thresh).astype(np.int32)  # [T, L]
    y = (y_vec > 0.5).astype(np.int32)                    # [L]

    # indices
    first_idx = _first_commit_idx(pred_sets)           # None if never commits
    final_idx = T - 1 if T > 0 else None

    # correctness at first commit & at final turn (any-overlap with GT)
    first_correct = False
    if first_idx is not None:
        first_correct = bool((pred_sets[first_idx] & y).any())
    final_correct = False if final_idx is None else bool((pred_sets[final_idx] & y).any())

    # classify trajectory
    if first_idx is None:
        category = "LATE"
    else:
        if first_correct and final_correct:
            category = "SC"       # stable correct
        elif first_correct and not final_correct:
            category = "REG"      # regression
        elif (not first_correct) and final_correct:
            category = "REC"      # recovery
        else:
            category = "SW"       # stuck wrong

    # timing
    first_correct_idx = _first_correct_idx(pred_sets, y)
    timing = earliness_and_horizon(turns, first_idx, first_correct_idx)

    # edit overhead over committed subsequence
    eo, total_changes, necessary, committed_flag = edit_overhead_sets(pred_sets, y)

    return dict(
        committed = committed_flag if first_idx is not None else False,
        category = category,
        first_idx = first_idx,
        final_idx = final_idx,
        first_correct = first_correct,
        final_correct = final_correct,
        commit_earliness = timing["commit_earliness"],
        commit_horizon   = timing["commit_horizon"],
        correct_earliness = timing["correct_earliness"],
        correct_horizon   = timing["correct_horizon"],
        eo = eo,
        eo_total_changes = total_changes,
        eo_necessary = necessary,
        T = int(turns[-1]) if T > 0 else None,
        first_turn = int(turns[first_idx]) if first_idx is not None else None,
        final_turn = int(turns[final_idx]) if final_idx is not None else None,
        first_correct_turn = (int(turns[first_correct_idx]) if first_correct_idx is not None else None),
    )

# ---------------------- corpus aggregation ----------------------

def aggregate(dialog_infos: List[Dict[str, Any]]) -> Dict[str, Any]:
    N = len(dialog_infos)
    cats = {}
    for di in dialog_infos:
        cats[di["category"]] = cats.get(di["category"], 0) + 1

    committed = sum(1 for di in dialog_infos if di["committed"])
    late = cats.get("LATE", 0)

    # First/Final accuracy on ALL dialogs (LATE has no first commit -> counted as not-first-correct)
    first_acc = (cats.get("SC", 0) + cats.get("REG", 0)) / N if N else 0.0
    final_acc = (cats.get("SC", 0) + cats.get("REC", 0)) / N if N else 0.0

    # Recovery / Regression (conditioned)
    denom_rec = (cats.get("REC", 0) + cats.get("SW", 0))
    recovery_rate = (cats.get("REC", 0) / denom_rec) if denom_rec else None

    denom_reg = (cats.get("REG", 0) + cats.get("SC", 0))
    regression_rate = (cats.get("REG", 0) / denom_reg) if denom_reg else None

    lateness = late / N if N else 0.0

    # Earliness / Horizon (means over dialogs where defined)
    commit_earl = [di["commit_earliness"] for di in dialog_infos if di["commit_earliness"] is not None]
    mean_commit_earliness = float(np.mean(commit_earl)) if commit_earl else None

    commit_hor = [di["commit_horizon"] for di in dialog_infos if di["commit_horizon"] is not None]
    mean_commit_horizon = float(np.mean(commit_hor)) if commit_hor else None

    correct_earl = [di["correct_earliness"] for di in dialog_infos if di["correct_earliness"] is not None]
    mean_correct_earliness = float(np.mean(correct_earl)) if correct_earl else None

    correct_hor = [di["correct_horizon"] for di in dialog_infos if di["correct_horizon"] is not None]
    mean_correct_horizon = float(np.mean(correct_hor)) if correct_hor else None

    # Edit overhead mean over dialogs where defined
    eos = [di["eo"] for di in dialog_infos if di["eo"] is not None]
    mean_eo = float(np.mean(eos)) if eos else None

    return {
        "N": N,
        "categories": cats,
        "first_accuracy": first_acc,
        "final_accuracy": final_acc,
        "recovery_rate": recovery_rate,
        "regression_rate": regression_rate,
        "lateness": lateness,
        "mean_commit_earliness": mean_commit_earliness,
        "mean_commit_horizon": mean_commit_horizon,
        "mean_correct_earliness": mean_correct_earliness,
        "mean_correct_horizon": mean_correct_horizon,
        "mean_edit_overhead": mean_eo,
        "num_committed": committed,
    }

# ---------------------- main: read NPZ and compute ----------------------

def load_turnwise_npz(path: str):
    Z = np.load(path, allow_pickle=True)
    files  = Z["files"]          # [N]
    turns  = Z["turn_idx"]       # [N]
    logits = Z["logits"]         # [N, L]
    labels = Z["labels"]         # [N, L]
    return files, turns, logits, labels

def evaluate_from_npz(npz_path: str, prob_thresh: float = 0.5, ece_bins: int = 15) -> Dict[str, Any]:
    files, turns, logits, labels = load_turnwise_npz(npz_path)
    dialogs = np.unique(files)

    dialog_infos: List[Dict[str, Any]] = []

    # for first/final metrics aggregation
    first_pred_sets, final_pred_sets = [], []
    first_labels,     final_labels   = [], []
    first_top1_hits,  final_top1_hits = [], []
    first_top1_rec_parts, final_top1_rec_parts = [], []

    # committed-dialog metrics
    first_commit_confs: List[float] = []
    final_commit_confs: List[float] = []
    dialog_ece_list: List[float] = []  # per-dialog ECE over committed subsequence

    for d in dialogs:
        m = (files == d)
        order = np.argsort(turns[m])                 # sort by turn
        t  = turns[m][order]                         # [T]
        lg = logits[m][order]                        # [T, L]
        y  = (labels[m][order][0] > 0.5).astype(np.int32)  # dialog GT [L]
        T  = t.size

        # probs & sets
        probs = sigmoid(lg)                          # [T, L]
        pred_sets = (probs >= prob_thresh).astype(np.int32)

        # indices
        on = np.where(pred_sets.sum(axis=1) > 0)[0]
        first_idx = int(on[0]) if on.size else None
        final_idx = T - 1 if T > 0 else None

        # collect per-dialog info (your existing struct)
        info = eval_dialog_from_npz_block(t, lg, y, prob_thresh)
        info["id"] = str(d)
        dialog_infos.append(info)

        # --------- collect FIRST sets & P@1/R@1 ----------
        if first_idx is None:
            # no commit -> empty set
            first_pred_sets.append(np.zeros_like(y))
            first_labels.append(y)
            # P@1/R@1: treat as miss
            gt_k = int(y.sum())
            first_top1_hits.append(0.0)
            first_top1_rec_parts.append(0.0 if gt_k > 0 else 0.0)
        else:
            first_pred_sets.append(pred_sets[first_idx])
            first_labels.append(y)
            # top1 computed from probabilities at first commit turn
            top1 = int(np.argmax(probs[first_idx]))
            hit = 1.0 if y[top1] == 1 else 0.0
            gt_k = int(y.sum())
            first_top1_hits.append(hit)
            first_top1_rec_parts.append((hit / gt_k) if gt_k > 0 else 0.0)

            # --------- confidence: first commit & final commit ----------
            last_commit_idx = int(on[-1])
            first_commit_confs.append(float(probs[first_idx].max()))
            final_commit_confs.append(float(probs[last_commit_idx].max()))

            # --------- per-dialog ECE over committed subsequence ----------
            probs_sub = probs[first_idx:]                     # [S, L]
            pred_sub  = pred_sets[first_idx:]                 # [S, L]
            conf_turn = probs_sub.max(axis=1)                 # [S]
            correct_turn = ((pred_sub & y).sum(axis=1) > 0)    # [S] bool (any-overlap correctness)
            dialog_ece_list.append(compute_ece(conf_turn, correct_turn, n_bins=ece_bins))

        # --------- collect FINAL sets & P@1/R@1 ----------
        if final_idx is None:
            # degenerate (no turns)
            final_pred_sets.append(np.zeros_like(y))
            final_labels.append(y)
            final_top1_hits.append(0.0)
            final_top1_rec_parts.append(0.0)
        else:
            final_pred_sets.append(pred_sets[final_idx])
            final_labels.append(y)
            top1_f = int(np.argmax(probs[final_idx]))
            hit_f = 1.0 if y[top1_f] == 1 else 0.0
            gt_k = int(y.sum())
            final_top1_hits.append(hit_f)
            final_top1_rec_parts.append((hit_f / gt_k) if gt_k > 0 else 0.0)

    # stack for metric computation
    first_pred_sets = np.stack(first_pred_sets, axis=0) if first_pred_sets else np.zeros((0, labels.shape[1]), dtype=np.int32)
    first_labels    = np.stack(first_labels, axis=0)    if first_labels    else np.zeros((0, labels.shape[1]), dtype=np.int32)
    final_pred_sets = np.stack(final_pred_sets, axis=0) if final_pred_sets else np.zeros((0, labels.shape[1]), dtype=np.int32)
    final_labels    = np.stack(final_labels, axis=0)    if final_labels    else np.zeros((0, labels.shape[1]), dtype=np.int32)

    # compute PR/F1
    first_stats = multilabel_prf1(first_pred_sets, first_labels)
    final_stats = multilabel_prf1(final_pred_sets, final_labels)

    # Precision@1 and Recall@1
    first_p1 = float(np.mean(first_top1_hits)) if len(first_top1_hits) else 0.0
    first_r1 = float(np.mean(first_top1_rec_parts)) if len(first_top1_rec_parts) else 0.0
    final_p1 = float(np.mean(final_top1_hits)) if len(final_top1_hits) else 0.0
    final_r1 = float(np.mean(final_top1_rec_parts)) if len(final_top1_rec_parts) else 0.0

    mean_first_commit_conf = float(np.mean(first_commit_confs)) if first_commit_confs else None
    mean_final_commit_conf = float(np.mean(final_commit_confs)) if final_commit_confs else None
    mean_dialog_ece_committed = float(np.mean(dialog_ece_list)) if dialog_ece_list else None

    # your existing corpus-level summary (cats, earliness, EO, etc.)
    summary_core = aggregate(dialog_infos)

    # extend summary with new metrics
    summary_core.update({
        # FIRST COMMIT metrics (set-based)
        "first_micro_f1": first_stats["micro_f1"],
        "first_macro_f1": first_stats["macro_f1"],
        "first_micro_p":  first_stats["micro_p"],
        "first_micro_r":  first_stats["micro_r"],
        "first_P@1": first_p1,
        "first_R@1": first_r1,

        # FINAL TURN metrics
        "final_micro_f1": final_stats["micro_f1"],
        "final_macro_f1": final_stats["macro_f1"],
        "final_micro_p":  final_stats["micro_p"],
        "final_micro_r":  final_stats["micro_r"],
        "final_P@1": final_p1,
        "final_R@1": final_r1,

        # threshold used
        "prob_threshold": float(prob_thresh),

        # confidence (over committed dialogs only)
        "mean_first_commit_conf": mean_first_commit_conf,
        "mean_final_commit_conf": mean_final_commit_conf,
        "num_committed_for_conf": int(len(first_commit_confs)),

        # per-dialog ECE averaged over committed dialogs only
        "mean_dialog_ece_committed": mean_dialog_ece_committed,
        "num_dialogs_ece_committed": int(len(dialog_ece_list)),
        "ece_bins": int(ece_bins),
    })

    return {"summary": summary_core, "per_dialog": dialog_infos}


# # ---------------------- CLI ----------------------
# if __name__ == "__main__":
#     ap = argparse.ArgumentParser("Turnwise Multi-label Trajectory Evaluator")
#     ap.add_argument("--npz", required=True, help="Path to turnwise NPZ (files, turn_idx, logits, labels)")
#     ap.add_argument("--thresh", type=float, default=0.5, help="Probability threshold for commit (sigmoid >= thresh)")
#     ap.add_argument("--ece_bins", type=int, default=15, help="Number of bins for ECE")
#     ap.add_argument("--dump_per_dialog", action="store_true", help="Print per-dialog JSON lines")
#     args = ap.parse_args()
#
#     out = evaluate_from_npz(args.npz, prob_thresh=args.thresh, ece_bins=args.ece_bins)
#     print("\n=== Corpus Summary ===")
#     print(json.dumps(out["summary"], indent=2))
#
#     if args.dump_per_dialog:
#         print("\n=== Per-dialog ===")
#         for di in out["per_dialog"]:
#             print(json.dumps(di, ensure_ascii=False))
