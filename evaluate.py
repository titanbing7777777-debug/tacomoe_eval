"""Utility for computing precision, recall, and F1 for dialogue extraction tasks.

The dataset stores one JSON object per line. Each object contains at least the
fields `sample_id`, `task_type`, and `target`. This script compares the gold
annotations against model predictions for all non-reply task types (targets,
aspects, opinions, quadruples, target-aspect, target-opinion, aspect-opinion).

Example:
    python evaluate.py --gold data/test.json --pred data/test_predictions.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

DEFAULT_TASKS: Sequence[str] = (
    "targets",
    "aspects",
    "opinions",
    "quadruples",
    "target-aspect",
    "target-opinion",
    "aspect-opinion",
)
SENTINEL_NON_OP = "statement-non-opinion"


def load_records(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load JSONL annotations keyed by sample_id."""
    records: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Failed to parse {path} line {lineno}: {exc}") from exc
            sample_id = obj.get("sample_id")
            if not sample_id:
                raise ValueError(f"Missing sample_id in {path} line {lineno}")
            if sample_id in records:
                raise ValueError(f"Duplicate sample_id '{sample_id}' in {path}")
            records[sample_id] = obj
    return records


def parse_target_field(raw: Optional[str]) -> Any:
    """Parse the serialized `target` field into Python data."""
    if raw is None:
        return []
    text = raw.strip()
    if not text:
        return []
    if text.lower() == SENTINEL_NON_OP:
        return SENTINEL_NON_OP
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def normalize_item(value: Any) -> Any:
    """Convert nested values into hashable, whitespace-trimmed counterparts."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return tuple(normalize_item(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, normalize_item(v)) for k, v in value.items()))
    return value


def items_from_entry(entry: Optional[Mapping[str, Any]]) -> Set[Any]:
    """Extract a set of normalized annotations from a record."""
    if not entry:
        return set()
    parsed = parse_target_field(entry.get("target"))
    if parsed == SENTINEL_NON_OP:
        return set()
    iterable: Iterable[Any]
    if isinstance(parsed, list):
        iterable = parsed
    elif parsed in (None, ""):
        iterable = []
    else:
        iterable = [parsed]
    return {normalize_item(item) for item in iterable if item not in (None, "")}


def collect_quadruples(records: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[Any]]:
    """Return all quadruple annotations keyed by sample id."""
    extracted: Dict[str, List[Any]] = {}
    for sample_id, entry in records.items():
        if entry.get("task_type") != "quadruples":
            continue
        items = sorted(items_from_entry(entry), key=str)
        extracted[sample_id] = [list(item) if isinstance(item, tuple) else item for item in items]
    return extracted


def dump_quadruples(gold_path: Path, pred_path: Path, output_path: Optional[Path] = None) -> None:
    """Print or save quadruple annotations from gold and predictions."""
    gold_records = load_records(gold_path)
    pred_records = load_records(pred_path)
    payload = {
        "gold": collect_quadruples(gold_records),
        "pred": collect_quadruples(pred_records),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        output_path.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


def update_counts(
    stats: MutableMapping[str, Counter],
    task_type: str,
    gold_items: Set[Any],
    pred_items: Set[Any],
) -> None:
    """Update true/false positive/negative counts for a task."""
    tp = len(gold_items & pred_items)
    fp = len(pred_items - gold_items)
    fn = len(gold_items - pred_items)
    stats[task_type]["tp"] += tp
    stats[task_type]["fp"] += fp
    stats[task_type]["fn"] += fn


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(stats: Mapping[str, Counter]) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, and F1 for each task and the micro-average."""
    metrics: Dict[str, Dict[str, float]] = {}
    total = Counter(tp=0, fp=0, fn=0)
    for task, counts in stats.items():
        total += counts
        precision = safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_div(counts["tp"], counts["tp"] + counts["fn"])
        f1 = safe_div(2 * precision * recall, precision + recall)
        metrics[task] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
        }
    precision = safe_div(total["tp"], total["tp"] + total["fp"])
    recall = safe_div(total["tp"], total["tp"] + total["fn"])
    f1 = safe_div(2 * precision * recall, precision + recall)
    metrics["overall"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total["tp"],
        "fp": total["fp"],
        "fn": total["fn"],
    }
    return metrics


def format_table(metrics: Mapping[str, Mapping[str, float]], task_order: Sequence[str]) -> str:
    """Render a text table of the metric dictionary."""
    header = f"{'Task':<15}{'TP':>8}{'FP':>8}{'FN':>8}{'P':>10}{'R':>10}{'F1':>10}"
    rows = [header, "-" * len(header)]
    ordered_tasks = list(task_order) + ["overall"]
    for task in ordered_tasks:
        data = metrics.get(task)
        if not data:
            continue
        rows.append(
            f"{task:<15}"
            f"{int(data['tp']):>8}"
            f"{int(data['fp']):>8}"
            f"{int(data['fn']):>8}"
            f"{data['precision']:>10.4f}"
            f"{data['recall']:>10.4f}"
            f"{data['f1']:>10.4f}"
        )
    return "\n".join(rows)


def evaluate(gold_path: Path, pred_path: Path, tasks: Sequence[str]) -> str:
    gold_records = load_records(gold_path)
    pred_records = load_records(pred_path)

    stats: Dict[str, Counter] = {task: Counter(tp=0, fp=0, fn=0) for task in tasks}

    for sample_id, gold in gold_records.items():
        task = gold.get("task_type")
        if task not in tasks:
            continue
        gold_items = items_from_entry(gold)
        pred_items = items_from_entry(pred_records.get(sample_id))
        update_counts(stats, task, gold_items, pred_items)

    for sample_id, pred in pred_records.items():
        task = pred.get("task_type")
        if task not in tasks or sample_id in gold_records:
            continue
        pred_items = items_from_entry(pred)
        stats[task]["fp"] += len(pred_items)

    metrics = compute_metrics(stats)
    return format_table(metrics, tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate dialogue extraction predictions.")
    parser.add_argument("--gold", type=Path, default=Path("data/test.json"), help="Path to gold JSONL file")
    parser.add_argument(
        "--pred",
        type=Path,
        default=Path("data/test_predictions.json"),
        help="Path to prediction JSONL file",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        choices=DEFAULT_TASKS,
        default=list(DEFAULT_TASKS),
        help="Subset of tasks to evaluate (defaults to all non-reply tasks)",
    )
    parser.add_argument(
        "--dump-quadruples",
        metavar="PATH",
        nargs="?",
        const="-",
        help="Export quadruple annotations to PATH (or stdout if omitted)",
    )
    args = parser.parse_args()

    if args.dump_quadruples is not None:
        output = None if args.dump_quadruples == "-" else Path(args.dump_quadruples)
        dump_quadruples(args.gold, args.pred, output)
        return

    table = evaluate(args.gold, args.pred, args.tasks)
    print(table)


if __name__ == "__main__":
    main()
