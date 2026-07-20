"""
Task loader -- decouples "what is a normal task" from session generation.

Reads every data/tasks/*.json file, validates the combined task set, and
exposes it as a plain list of dicts. lgnn_experiment.py (session generator)
consumes this instead of hardcoding task prompts inline, so the reference
dataset for the normal class has an explicit, auditable, versioned source
independent of the collection code.

Pipeline (see README §정상 task source 객관화):
    task JSON files  ->  task_loader (this file)  ->  split manifest
    (generate_task_split.py)  ->  session generator (lgnn_experiment.py)
    ->  Ollama execution  ->  metadata dataset

Required per-task schema (see data/tasks/*.json):
    task_id, category, prompt, source_type, source_name, source_item_id,
    license, notes
"""
import json
import os

TASKS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "tasks")
MIN_CATEGORIES = 5
REQUIRED_FIELDS = ["task_id", "category", "prompt", "source_type",
                   "source_name", "source_item_id", "license", "notes"]


def load_all_tasks(tasks_dir=TASKS_DIR):
    """
    Loads and validates every *.json file in tasks_dir. Each file is expected
    to hold a JSON list of task dicts (one file per category, by convention,
    but validation only checks the combined set -- it doesn't require a
    1:1 file<->category mapping).

    Validates:
      - every task has all REQUIRED_FIELDS (missing key -> AssertionError)
      - task_id is globally unique across all files
      - task["category"] matches the file's declared category for every task
        in that file (catches copy-paste mistakes across category files)
      - at least MIN_CATEGORIES distinct categories exist in total

    Returns a flat list of task dicts, in file-then-in-file order (stable,
    not shuffled -- shuffling/splitting is generate_task_split.py's job).
    """
    assert os.path.isdir(tasks_dir), f"tasks directory not found: {tasks_dir}"

    all_tasks = []
    seen_ids = {}
    for fname in sorted(os.listdir(tasks_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(tasks_dir, fname)
        with open(path, encoding="utf-8") as f:
            tasks = json.load(f)
        assert isinstance(tasks, list), f"{fname}: expected a JSON list of task objects"

        file_category = os.path.splitext(fname)[0]
        for t in tasks:
            missing = [k for k in REQUIRED_FIELDS if k not in t]
            assert not missing, f"{fname}/{t.get('task_id', '?')}: missing field(s) {missing}"
            assert t["category"] == file_category, \
                f"{fname}: task {t['task_id']} has category={t['category']!r}, " \
                f"expected {file_category!r} (file/category mismatch)"
            assert t["task_id"] not in seen_ids, \
                f"duplicate task_id {t['task_id']!r} in {fname} (already seen in {seen_ids[t['task_id']]})"
            seen_ids[t["task_id"]] = fname
            all_tasks.append(t)

    categories = sorted({t["category"] for t in all_tasks})
    assert len(categories) >= MIN_CATEGORIES, \
        f"need at least {MIN_CATEGORIES} task categories, found {len(categories)}: {categories}"

    return all_tasks


def category_counts(tasks):
    counts = {}
    for t in tasks:
        counts[t["category"]] = counts.get(t["category"], 0) + 1
    return dict(sorted(counts.items()))


def print_category_counts(tasks):
    counts = category_counts(tasks)
    print(f"  Loaded {len(tasks)} tasks across {len(counts)} categories:")
    for cat, n in counts.items():
        print(f"    {cat:<22} {n:3d}")


def tasks_by_id(tasks):
    return {t["task_id"]: t for t in tasks}


if __name__ == "__main__":
    tasks = load_all_tasks()
    print_category_counts(tasks)
    print(f"\n  All task_id values unique: {len(tasks) == len(tasks_by_id(tasks))}")
    print(f"  Sample task: {tasks[0]}")
