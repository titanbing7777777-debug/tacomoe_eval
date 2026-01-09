# Dialogue Task Evaluation

`evaluate.py` compares gold annotations against model predictions for the non-reply dialogue tasks (targets, aspects, opinions, quadruples, target-aspect, target-opinion, and aspect-opinion). The inputs are JSONL files where each line is a dictionary that includes `sample_id`, `task_type`, and `target`.

## Usage

```bash
D:/Code/eval/.venv/Scripts/python.exe evaluate.py \
    --gold data/test.json \
    --pred data/test_predictions.json
```

Key points:
- By default the script evaluates every non-reply task. Use `--tasks` to restrict the set, e.g. `--tasks targets opinions`.
- The loader expects newline-delimited JSON (one object per line). Empty lines are ignored.
- All matching logic is set-based, so duplicate predictions for the same item are counted once.

### Dumping quadruples only

If you simply want the quadruple annotations for manual comparison, use:

```bash
D:/Code/eval/.venv/Scripts/python.exe evaluate.py --dump-quadruples quadruples.json
```

When `--dump-quadruples` is provided without a path it prints to stdout. The produced JSON contains two dictionaries (`gold` and `pred`) that map each `sample_id` to the list of normalized quadruples.

## Output

The script prints a table with per-task and overall true positives, false positives, false negatives, and the derived precision/recall/F1 scores. Example (truncated):

```
Task                 TP      FP      FN         P         R        F1
---------------------------------------------------------------------
targets             661     106     101    0.8618    0.8675    0.8646
...
overall            2732    1876    2164    0.5929    0.5580    0.5749
```

Use the numbers to track progress across model iterations or to compare the quality of different prediction files.
