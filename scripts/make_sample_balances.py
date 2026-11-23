"""Generate a 24-month balance sheet consistent with sample_financials.csv.

Receivables ~ 30 days of revenue, payables ~ 20 days of cash costs, 2k/month
of capex, flat debt and deposits. Cash and equity follow from the income
statement, so every month balances.
"""
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine import load_csv, to_monthly  # noqa: E402

DAYS = 365 / 12
OUT = ROOT / "sample_data" / "sample_balances.csv"


def main():
    random.seed(11)
    m = to_monthly(load_csv(ROOT / "sample_data" / "sample_financials.csv"))
    cash, fa, debt, deposits, accrued = 180_000.0, 120_000.0, 150_000.0, 20_000.0, 30_000.0
    ar_prev = ap_prev = None
    rows = []
    for i, (period, r) in enumerate(m.iterrows()):
        ar = r.revenue * 30 / DAYS * (1 + random.uniform(-0.05, 0.05))
        ap = (r.cogs + r.opex - r.depreciation) * 20 / DAYS * (1 + random.uniform(-0.05, 0.05))
        capex = 2_000.0
        net_income = r.revenue - r.cogs - r.opex + r.other
        if i > 0:  # cash moves with the cash flow; month 0 is the starting point
            cash += net_income + r.depreciation - (ar - ar_prev) + (ap - ap_prev) - capex
            fa += capex - r.depreciation
        ar_prev, ap_prev = ar, ap
        assets = cash + ar + fa + deposits
        equity = assets - ap - debt - accrued
        d = period.date().isoformat()
        rows += [
            (d, "Bank accounts", "cash", cash), (d, "Accounts receivable", "receivables", ar),
            (d, "Equipment (net)", "fixed_assets", fa), (d, "Deposits", "other_assets", deposits),
            (d, "Accounts payable", "payables", ap), (d, "Term loan", "debt", debt),
            (d, "Accrued expenses", "other_liability", accrued), (d, "Shareholders' equity", "equity", equity),
        ]
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "account", "category", "amount"])
        w.writerows((d, a, c, round(v, 2)) for d, a, c, v in rows)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
