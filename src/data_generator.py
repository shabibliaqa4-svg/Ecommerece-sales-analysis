"""
Generates a realistic e-commerce sales dataset with 50,000 rows and 18 columns.
Includes intentional data quality issues (nulls, duplicates, inconsistent
formatting) so the cleaning step is meaningful.

Run:  python src/data_generator.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
NUM_ROWS = 50_000
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "ecommerce_sales_raw.csv",
)

REGIONS = ["West", "East", "South", "Central"]
REGION_WEIGHTS = [0.30, 0.28, 0.22, 0.20]

CATEGORIES = ["Electronics", "Clothing", "Home & Garden",
              "Sports", "Books", "Beauty"]
CATEGORY_WEIGHTS = [0.25, 0.22, 0.18, 0.15, 0.10, 0.10]

PRODUCTS = {
    "Electronics":   ["Laptop", "Smartphone", "Tablet", "Headphones",
                      "Smartwatch", "Camera", "Speaker", "Monitor"],
    "Clothing":      ["T-Shirt", "Jeans", "Jacket", "Sneakers",
                      "Dress", "Hoodie", "Shorts", "Sweater"],
    "Home & Garden": ["Lamp", "Rug", "Planter", "Cookware Set",
                      "Bed Sheets", "Vacuum", "Blender", "Candle Set"],
    "Sports":        ["Yoga Mat", "Dumbbells", "Running Shoes",
                      "Bicycle", "Tennis Racket", "Football",
                      "Fitness Tracker", "Backpack"],
    "Books":         ["Fiction Novel", "Cookbook", "Self-Help Book",
                      "Biography", "Science Textbook", "Art Book",
                      "Children's Book", "Travel Guide"],
    "Beauty":        ["Perfume", "Skincare Set", "Lipstick",
                      "Hair Dryer", "Nail Kit", "Face Mask",
                      "Moisturizer", "Sunscreen"],
}

PRICE_RANGES = {
    "Electronics":   (49.99, 1299.99),
    "Clothing":      (14.99, 199.99),
    "Home & Garden": (9.99, 349.99),
    "Sports":        (12.99, 499.99),
    "Books":         (7.99, 59.99),
    "Beauty":        (8.99, 149.99),
}

CUSTOMER_SEGMENTS = ["Regular", "Premium", "New"]
SEGMENT_WEIGHTS = [0.50, 0.18, 0.32]

PAYMENT_METHODS = ["Credit Card", "PayPal", "Debit Card", "Cash on Delivery"]
PAYMENT_WEIGHTS = [0.35, 0.28, 0.22, 0.15]

SHIPPING_TYPES = ["Standard", "Express", "Same-Day"]
SHIPPING_WEIGHTS = [0.55, 0.30, 0.15]

ORDER_STATUSES = ["Delivered", "Returned", "Cancelled", "Processing"]
STATUS_WEIGHTS = [0.82, 0.08, 0.05, 0.05]

# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────

def generate_dataset() -> pd.DataFrame:
    """Build the raw dataset with realistic patterns and deliberate
    data-quality issues."""

    rng = np.random.default_rng(SEED)

    # ── Order dates (with seasonality) ────────────────────────────────────
    start = datetime(2022, 1, 1)
    end = datetime(2024, 12, 31)
    total_days = (end - start).days

    # Weight each month to create seasonality (peaks in summer & holidays)
    month_weights = {
        1: 0.7, 2: 0.65, 3: 0.8, 4: 0.85, 5: 0.95,
        6: 1.15, 7: 1.20, 8: 1.10, 9: 0.90, 10: 0.95,
        11: 1.30, 12: 1.40,
    }

    dates = []
    while len(dates) < NUM_ROWS:
        day_offset = rng.integers(0, total_days)
        candidate = start + timedelta(days=int(day_offset))
        weight = month_weights[candidate.month]
        if rng.random() < weight / 1.40:   # accept/reject sampling
            dates.append(candidate)
    dates = sorted(dates[:NUM_ROWS])

    # ── Core columns ──────────────────────────────────────────────────────
    order_ids = [f"ORD-{100000 + i}" for i in range(NUM_ROWS)]
    customer_ids = [f"CUST-{rng.integers(10000, 30000)}" for _ in range(NUM_ROWS)]

    categories = rng.choice(CATEGORIES, size=NUM_ROWS, p=CATEGORY_WEIGHTS)
    products = [rng.choice(PRODUCTS[cat]) for cat in categories]

    regions = rng.choice(REGIONS, size=NUM_ROWS, p=REGION_WEIGHTS)
    segments = rng.choice(CUSTOMER_SEGMENTS, size=NUM_ROWS, p=SEGMENT_WEIGHTS)
    payments = rng.choice(PAYMENT_METHODS, size=NUM_ROWS, p=PAYMENT_WEIGHTS)
    shipping = rng.choice(SHIPPING_TYPES, size=NUM_ROWS, p=SHIPPING_WEIGHTS)
    statuses = rng.choice(ORDER_STATUSES, size=NUM_ROWS, p=STATUS_WEIGHTS)

    # ── Numeric columns ──────────────────────────────────────────────────
    quantities = rng.integers(1, 6, size=NUM_ROWS)

    unit_prices = np.array([
        round(rng.uniform(*PRICE_RANGES[cat]), 2) for cat in categories
    ])

    # Premium customers get smaller discounts; New customers get welcome deals
    discount_base = rng.choice([0, 5, 10, 15, 20, 25, 30],
                               size=NUM_ROWS,
                               p=[0.30, 0.20, 0.18, 0.12, 0.10, 0.06, 0.04])
    discount_pct = discount_base.astype(float)

    revenue = np.round(quantities * unit_prices * (1 - discount_pct / 100), 2)
    cost_ratio = rng.uniform(0.40, 0.70, size=NUM_ROWS)
    cost = np.round(revenue * cost_ratio, 2)
    profit = np.round(revenue - cost, 2)

    ratings = np.round(rng.normal(loc=3.8, scale=0.8, size=NUM_ROWS), 1)
    ratings = np.clip(ratings, 1.0, 5.0)

    # ── Build DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame({
        "order_id":          order_ids,
        "order_date":        dates,
        "customer_id":       customer_ids,
        "customer_segment":  segments,
        "region":            regions,
        "product_category":  categories,
        "product_name":      products,
        "quantity":          quantities,
        "unit_price":        unit_prices,
        "discount_pct":      discount_pct,
        "revenue":           revenue,
        "cost":              cost,
        "profit":            profit,
        "payment_method":    payments,
        "shipping_type":     shipping,
        "order_status":      statuses,
        "customer_rating":   ratings,
    })

    # ── Inject data-quality issues ───────────────────────────────────────

    # 1. ~3 % missing in customer_rating
    mask_rating = rng.random(NUM_ROWS) < 0.03
    df.loc[mask_rating, "customer_rating"] = np.nan

    # 2. ~2 % missing in discount_pct
    mask_disc = rng.random(NUM_ROWS) < 0.02
    df.loc[mask_disc, "discount_pct"] = np.nan

    # 3. ~1.5 % missing in region
    mask_region = rng.random(NUM_ROWS) < 0.015
    df.loc[mask_region, "region"] = np.nan

    # 4. ~1 % missing in shipping_type
    mask_ship = rng.random(NUM_ROWS) < 0.01
    df.loc[mask_ship, "shipping_type"] = np.nan

    # 5. Inconsistent category casing
    random_rows = rng.choice(NUM_ROWS, size=500, replace=False)
    df.loc[random_rows, "product_category"] = (
        df.loc[random_rows, "product_category"].str.upper()
    )
    random_rows2 = rng.choice(NUM_ROWS, size=300, replace=False)
    df.loc[random_rows2, "product_category"] = (
        df.loc[random_rows2, "product_category"].str.lower()
    )

    # 6. Inconsistent region naming
    df.loc[rng.choice(NUM_ROWS, size=200, replace=False), "region"] = "west"
    df.loc[rng.choice(NUM_ROWS, size=150, replace=False), "region"] = "EAST"
    df.loc[rng.choice(NUM_ROWS, size=100, replace=False), "region"] = "south "

    # 7. Duplicate rows (~1 %)
    dup_idx = rng.choice(NUM_ROWS, size=500, replace=False)
    duplicates = df.iloc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # 8. Negative revenue on some returned / cancelled orders
    returned_mask = df["order_status"].isin(["Returned", "Cancelled"])
    flip_idx = df[returned_mask].sample(frac=0.3, random_state=SEED).index
    df.loc[flip_idx, "revenue"] = -abs(df.loc[flip_idx, "revenue"])
    df.loc[flip_idx, "profit"] = -abs(df.loc[flip_idx, "profit"])

    # 9. A few wrong data types (dates stored as strings already via CSV)
    df["order_date"] = df["order_date"].astype(str)

    # Shuffle
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Dataset generated: {OUTPUT_PATH}")
    print(f"   Rows   : {len(dataset):,}")
    print(f"   Columns: {dataset.shape[1]}")
    print(f"\nFirst 5 rows:\n{dataset.head()}")
    print(f"\nColumn types:\n{dataset.dtypes}")
    print(f"\nNull counts:\n{dataset.isnull().sum()}")
