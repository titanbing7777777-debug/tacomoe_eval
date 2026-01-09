"""逐条浏览 gold/pred 四元组结果的辅助脚本。"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence


def load_dump(path: Path) -> Mapping[str, Mapping[str, List[List[str]]]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(items: Sequence) -> str:
    if not items:
        return "[]"
    return json.dumps(items, ensure_ascii=False)


def browse(gold: Mapping[str, List], pred: Mapping[str, List]) -> None:
    keys = sorted(set(gold) | set(pred))
    total = len(keys)
    for idx, sample_id in enumerate(keys, start=1):
        gold_items = gold.get(sample_id, [])
        pred_items = pred.get(sample_id, [])
        print("-" * 80)
        print(f"[{idx}/{total}] sample_id: {sample_id}")
        print(f"gold: {summarize(gold_items)}")
        print(f"pred: {summarize(pred_items)}")
        user_input = input("回车继续，输入 q 退出: ").strip().lower()
        if user_input == "q":
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="逐条查看四元组 gold/pred")
    parser.add_argument(
        "path",
        type=Path,
        default=Path("quadruples_dump.json"),
        nargs="?",
        help="dump_quadruples 生成的 JSON 文件路径",
    )
    args = parser.parse_args()

    data = load_dump(args.path)
    gold = data.get("gold", {})
    pred = data.get("pred", {})
    browse(gold, pred)


if __name__ == "__main__":
    main()