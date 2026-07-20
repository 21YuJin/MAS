"""
Generates the fixed train/validation/test split over TASK IDs (not sessions)
and saves it to data/splits/normal_task_split_v1.json.

Run ONCE. The split is saved and then loaded read-only by every downstream
script (session generator, lgnn_experiment.py) -- regenerating it on every run
would silently change which tasks are "test" tasks between runs and make
results incomparable. Re-running this script is a no-op (prints and exits)
unless --force is passed, which requires an explicit --reason.

Split is stratified per category so every category is represented
proportionally in every split (10 tasks/category x 60/20/20 = 6/2/2 exactly,
no rounding needed at the current 50-task scale). Because the split is by
task_id, all sessions later generated from the same task (repeat runs,
paraphrases) automatically land in the same split -- there is no way for a
task to appear on both sides of train/test once this file is fixed.
"""
import argparse
import json
import os
import random

from task_loader import load_all_tasks, category_counts, print_category_counts

SPLITS_DIR    = os.path.join(os.path.dirname(__file__), "..", "..", "data", "splits")
SPLIT_PATH    = os.path.join(SPLITS_DIR, "normal_task_split_v1.json")
SPLIT_VERSION = "normal_task_split_v1"
SPLIT_SEED    = 2026   # fixed -- changing this changes the split, hence the version suffix above
SPLIT_FRACTIONS = {"train": 0.60, "validation": 0.20, "test": 0.20}


def stratified_task_split(tasks, seed=SPLIT_SEED, fractions=SPLIT_FRACTIONS):
    """
    Splits task_ids into train/validation/test, stratified per category so
    each category's proportion is preserved in every split. Within each
    category, tasks are shuffled with `seed` before slicing, so which specific
    tasks land in which split is deterministic but not alphabetical.
    """
    by_cat = {}
    for t in tasks:
        by_cat.setdefault(t["category"], []).append(t["task_id"])

    out = {"train": [], "validation": [], "test": []}
    for cat, ids in sorted(by_cat.items()):
        ids = sorted(ids)                      # deterministic starting order
        random.Random(seed).shuffle(ids)        # deterministic shuffle
        n = len(ids)
        n_tr  = round(n * fractions["train"])
        n_val = round(n * fractions["validation"])
        n_te  = n - n_tr - n_val                # remainder absorbs rounding
        out["train"].extend(ids[:n_tr])
        out["validation"].extend(ids[n_tr:n_tr + n_val])
        out["test"].extend(ids[n_tr + n_val:])

    for k in out:
        out[k] = sorted(out[k])
    return out


def validate_split(split, all_task_ids):
    train, val, test = set(split["train"]), set(split["validation"]), set(split["test"])
    assert not (train & val),  "train/validation share task_id(s) -- split is broken"
    assert not (train & test), "train/test share task_id(s) -- split is broken"
    assert not (val & test),   "validation/test share task_id(s) -- split is broken"
    union = train | val | test
    assert union == set(all_task_ids), \
        f"split does not partition every task_id exactly once (missing={set(all_task_ids)-union}, extra={union-set(all_task_ids)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                     help="overwrite an existing split file (requires --reason)")
    ap.add_argument("--reason", default=None,
                     help="required with --force: why the fixed split is being regenerated")
    args = ap.parse_args()

    if os.path.exists(SPLIT_PATH) and not args.force:
        print(f"[SKIP] {SPLIT_PATH} already exists. The split is generated once and reused --")
        print("       pass --force --reason \"...\" if you really intend to replace it")
        print("       (this changes which tasks are train/val/test for every future run).")
        with open(SPLIT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"       existing split: train={len(existing['train_task_ids'])}  "
              f"val={len(existing['validation_task_ids'])}  test={len(existing['test_task_ids'])}")
        return

    if args.force and not args.reason:
        raise SystemExit("--force requires --reason \"...\" (recorded in the output file)")

    os.makedirs(SPLITS_DIR, exist_ok=True)

    tasks = load_all_tasks()
    print_category_counts(tasks)

    split = stratified_task_split(tasks)
    all_ids = [t["task_id"] for t in tasks]
    validate_split(split, all_ids)

    cat_by_id = {t["task_id"]: t["category"] for t in tasks}
    per_split_category_counts = {
        name: category_counts([{"category": cat_by_id[tid]} for tid in ids])
        for name, ids in split.items()
    }

    payload = {
        "split_version": SPLIT_VERSION,
        "split_seed": SPLIT_SEED,
        "split_fractions": SPLIT_FRACTIONS,
        "n_tasks_total": len(tasks),
        "train_task_ids": split["train"],
        "validation_task_ids": split["validation"],
        "test_task_ids": split["test"],
        "per_split_category_counts": per_split_category_counts,
        "regenerated_reason": args.reason if args.force else None,
    }
    with open(SPLIT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  [saved] {SPLIT_PATH}")
    print(f"  train={len(split['train'])}  validation={len(split['validation'])}  test={len(split['test'])}")
    for name in ("train", "validation", "test"):
        print(f"    {name:<10} {per_split_category_counts[name]}")


if __name__ == "__main__":
    main()
