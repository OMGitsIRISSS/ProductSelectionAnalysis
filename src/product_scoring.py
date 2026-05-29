from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sample_products.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "top_products.csv"


POSITIVE_METRICS = [
    "monthly_sales",
    "repurchase_rate",
    "gross_margin",
    "rating",
    "fulfillment_48h_rate",
    "scenario_fit_score",
]

NEGATIVE_METRICS = ["return_rate"]

WEIGHTS = {
    "monthly_sales": 0.25,
    "repurchase_rate": 0.20,
    "gross_margin": 0.20,
    "fulfillment_48h_rate": 0.15,
    "rating": 0.10,
    "scenario_fit_score": 0.10,
    "return_rate": -0.10,
}


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Scale a numeric column to 0-1 range."""
    min_value = series.min()
    max_value = series.max()
    if max_value == min_value:
        return pd.Series(0.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def score_products(products: pd.DataFrame) -> pd.DataFrame:
    scored = products.copy()

    for metric in POSITIVE_METRICS + NEGATIVE_METRICS:
        scored[f"{metric}_norm"] = min_max_normalize(scored[metric])

    scored["selection_score"] = 0.0
    for metric, weight in WEIGHTS.items():
        scored["selection_score"] += scored[f"{metric}_norm"] * weight

    return scored.sort_values("selection_score", ascending=False)


def summarize_by_category(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby("category")
        .agg(
            product_count=("product_id", "count"),
            avg_selection_score=("selection_score", "mean"),
            avg_monthly_sales=("monthly_sales", "mean"),
            avg_repurchase_rate=("repurchase_rate", "mean"),
            avg_gross_margin=("gross_margin", "mean"),
            avg_return_rate=("return_rate", "mean"),
        )
        .sort_values("avg_selection_score", ascending=False)
        .round(3)
    )


def main() -> None:
    products = pd.read_csv(DATA_PATH)
    scored = score_products(products)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.head(10).to_csv(OUTPUT_PATH, index=False)

    print("Top 10 product recommendations:")
    print(
        scored[
            [
                "product_id",
                "product_name",
                "category",
                "monthly_sales",
                "repurchase_rate",
                "gross_margin",
                "return_rate",
                "selection_score",
            ]
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    print("\nCategory-level summary:")
    print(summarize_by_category(scored).to_string())
    print(f"\nSaved top product list to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
