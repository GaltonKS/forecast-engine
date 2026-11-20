"""Generate 24 months of realistic-looking sample data for a small SaaS company."""
import csv
import random
from pathlib import Path

random.seed(7)
OUT = Path(__file__).resolve().parent.parent / "sample_data" / "sample_financials.csv"

ACCOUNTS = [
    # account, category, starting amount, monthly growth, noise
    ("Subscription revenue", "revenue", 80_000, 0.035, 0.03),
    ("Services revenue", "revenue", 15_000, 0.01, 0.10),
    ("Hosting", "cogs", 14_000, 0.03, 0.04),
    ("Payment processing", "cogs", 2_800, 0.033, 0.03),
    ("Salaries", "opex", 55_000, 0.015, 0.01),
    ("Rent", "opex", 6_000, 0.0, 0.0),
    ("Marketing", "opex", 12_000, 0.02, 0.15),
    ("Interest income", "other", 400, 0.0, 0.05),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period", "account", "category", "amount"])
        for m in range(24):
            year, month = 2024 + (m // 12), (m % 12) + 1
            for name, cat, start, g, noise in ACCOUNTS:
                amt = start * (1 + g) ** m * (1 + random.uniform(-noise, noise))
                w.writerow([f"{year}-{month:02d}-01", name, cat, round(amt, 2)])
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
