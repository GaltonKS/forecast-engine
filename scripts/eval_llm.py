"""Score a model endpoint on llm/evals.json (prompt -> expected scenario).

    LLM_BASE_URL=https://.../v1 LLM_MODEL=Qwen/Qwen2.5-7B-Instruct python scripts/eval_llm.py
    python scripts/eval_llm.py --fake          # self-check with the stand-in model
    python scripts/eval_llm.py --only growth-halves --json report.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import load_csv, to_monthly  # noqa: E402
from llm.client import LLMConfig  # noqa: E402
from llm.evals import SAMPLE, fake_transport, load_cases, run_case  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="use the stand-in model (no endpoint needed)")
    ap.add_argument("--only", action="append", help="case id(s) to run")
    ap.add_argument("--json", help="write the full report here")
    args = ap.parse_args()

    cfg = LLMConfig.from_env()
    if args.fake:
        cfg = LLMConfig(base_url="http://fake/v1", model="fake", transport=fake_transport())
    if not cfg.configured:
        sys.exit("Set LLM_BASE_URL and LLM_MODEL (or use --fake).")

    monthly = to_monthly(load_csv(SAMPLE))
    cases = [c for c in load_cases() if not args.only or c["id"] in args.only]
    results = []
    for c in cases:
        r = run_case(c, monthly, cfg)
        results.append(r)
        mark = "PASS" if r["passed"] else "FAIL"
        lat = f"{r['latency_seconds']:.1f}s" if r["latency_seconds"] is not None else "-"
        print(f"{mark}  {c['id']:20s} {lat:>7s}  {c['prompt']}")
        for f in r["failures"]:
            print(f"        - {f}")
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} passed  (model: {cfg.model})")
    if args.json:
        Path(args.json).write_text(json.dumps({"model": cfg.model, "results": results}, indent=1))


if __name__ == "__main__":
    main()
