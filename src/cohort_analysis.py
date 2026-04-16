"""
Cohort Analysis module - customer retention, lifetime value, and cohort behavior tracking.
"""

import numpy as np
import pandas as pd
from datetime import datetime


class CohortAnalyzer:
    """Performs cohort analysis on customer data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["order_date"] = pd.to_datetime(self.df["order_date"])
        self.cohorts: pd.DataFrame = pd.DataFrame()

    # ──────────────────────────────────────────────────────────────────────
    # Cohort Creation
    # ──────────────────────────────────────────────────────────────────────

    def create_cohorts(self) -> pd.DataFrame:
        """Create customer cohorts based on first purchase month."""
        # Get first purchase date per customer
        first_purchase = (
            self.df
            .groupby("customer_id")["order_date"]
            .min()
            .reset_index()
            .rename(columns={"order_date": "cohort_date"})
        )

        # Extract cohort month
        first_purchase["cohort_month"] = (
            first_purchase["cohort_date"]
            .dt.to_period("M")
            .astype(str)
        )

        # Merge cohort info back to original data
        df_with_cohort = self.df.merge(
            first_purchase[["customer_id", "cohort_month"]],
            on="customer_id",
            how="left"
        )

        # Calculate customer age (months since cohort)
        df_with_cohort["order_month"] = (
            df_with_cohort["order_date"]
            .dt.to_period("M")
            .astype(str)
        )

        df_with_cohort["months_since_cohort"] = (
            (df_with_cohort["order_date"].dt.to_period("M").astype(int) -
             pd.Period(df_with_cohort["cohort_month"], freq="M").astype(int))
        )

        self.cohorts = df_with_cohort
        return df_with_cohort

    # ──────────────────────────────────────────────────────────────────────
    # Retention Analysis
    # ──────────────────────────────────────────────────────────────────────

    def retention_table(self) -> pd.DataFrame:
        """Create cohort retention table (% of customers making repeat purchases)."""
        if self.cohorts.empty:
            self.create_cohorts()

        # Count unique customers per cohort-age bucket
        cohort_data = (
            self.cohorts
            .groupby(["cohort_month", "months_since_cohort"])["customer_id"]
            .nunique()
            .reset_index()
            .rename(columns={"customer_id": "unique_customers"})
        )

        # Get cohort sizes (month 0)
        cohort_sizes = cohort_data[cohort_data["months_since_cohort"] == 0][
            ["cohort_month", "unique_customers"]
        ].copy()
        cohort_sizes.rename(columns={"unique_customers": "cohort_size"}, inplace=True)

        # Merge and calculate retention rate
        retention = cohort_data.merge(cohort_sizes, on="cohort_month")
        retention["retention_rate"] = (
            retention["unique_customers"] / retention["cohort_size"] * 100
        ).round(1)

        # Pivot to create retention matrix
        retention_matrix = retention.pivot(
            index="cohort_month",
            columns="months_since_cohort",
            values="retention_rate"
        )

        return retention_matrix

    def retention_table_absolute(self) -> pd.DataFrame:
        """Create cohort retention table with absolute customer counts."""
        if self.cohorts.empty:
            self.create_cohorts()

        cohort_data = (
            self.cohorts
            .groupby(["cohort_month", "months_since_cohort"])["customer_id"]
            .nunique()
            .reset_index()
        )

        retention_matrix = cohort_data.pivot(
            index="cohort_month",
            columns="months_since_cohort",
            values="customer_id"
        )

        return retention_matrix

    # ──────────────────────────────────────────────────────────────────────
    # Revenue by Cohort (Lifetime Value by Cohort)
    # ──────────────────────────────────────────────────────────────────────

    def cohort_revenue(self) -> pd.DataFrame:
        """Revenue generated per cohort over time."""
        if self.cohorts.empty:
            self.create_cohorts()

        # Filter to delivered orders only
        delivered = self.cohorts[self.cohorts["order_status"] == "Delivered"].copy()

        cohort_revenue = (
            delivered
            .groupby(["cohort_month", "months_since_cohort"])["revenue"]
            .sum()
            .reset_index()
        )

        revenue_matrix = cohort_revenue.pivot(
            index="cohort_month",
            columns="months_since_cohort",
            values="revenue"
        ).round(0)

        return revenue_matrix

    def avg_order_value_by_cohort(self) -> pd.DataFrame:
        """Average order value by cohort over time."""
        if self.cohorts.empty:
            self.create_cohorts()

        delivered = self.cohorts[self.cohorts["order_status"] == "Delivered"].copy()

        aov = (
            delivered
            .groupby(["cohort_month", "months_since_cohort"])["revenue"]
            .mean()
            .reset_index()
        )

        aov_matrix = aov.pivot(
            index="cohort_month",
            columns="months_since_cohort",
            values="revenue"
        ).round(2)

        return aov_matrix

    # ──────────────────────────────────────────────────────────────────────
    # Customer Lifetime Value
    # ──────────────────────────────────────────────────────────────────────

    def customer_lifetime_value(self) -> pd.DataFrame:
        """Calculate CLV per customer."""
        if self.cohorts.empty:
            self.create_cohorts()

        delivered = self.cohorts[self.cohorts["order_status"] == "Delivered"].copy()

        clv = (
            delivered
            .groupby("customer_id")
            .agg(
                total_revenue=("revenue", "sum"),
                total_orders=("order_id", "count"),
                avg_order_value=("revenue", "mean"),
                months_active=(
                    "months_since_cohort",
                    lambda x: (x.max() - x.min()) + 1 if len(x) > 0 else 0
                ),
                first_purchase=("cohort_month", "first"),
                avg_rating=("customer_rating", "mean"),
            )
            .reset_index()
            .round(2)
        )

        clv = clv.sort_values("total_revenue", ascending=False)
        return clv

    def clv_by_segment(self) -> pd.DataFrame:
        """CLV analysis by customer segment."""
        if self.cohorts.empty:
            self.create_cohorts()

        delivered = self.cohorts[self.cohorts["order_status"] == "Delivered"].copy()

        clv_seg = (
            delivered
            .groupby("customer_segment")
            .agg(
                avg_clv=("revenue", lambda x: x.sum() / delivered["customer_id"].nunique()),
                total_customers=("customer_id", "nunique"),
                total_revenue=("revenue", "sum"),
                avg_order_value=("revenue", "mean"),
                repeat_purchase_rate=(
                    "order_id",
                    lambda x: len(x) / delivered["customer_id"].nunique()
                ),
                avg_rating=("customer_rating", "mean"),
            )
            .round(2)
        )

        return clv_seg

    # ──────────────────────────────────────────────────────────────────────
    # Cohort Insights
    # ──────────────────────────────────────────────────────────────────────

    def cohort_insights(self) -> dict:
        """Generate key insights from cohort analysis."""
        if self.cohorts.empty:
            self.create_cohorts()

        retention = self.retention_table()
        revenue = self.cohort_revenue()
        clv = self.customer_lifetime_value()
        clv_seg = self.clv_by_segment()

        insights = {}

        # 1. Best performing cohort
        best_cohort = revenue.sum(axis=1).idxmax()
        best_revenue = revenue.sum(axis=1).max()
        insights["best_cohort"] = f"{best_cohort} with ${best_revenue:,.0f} total revenue"

        # 2. Retention trend
        month_0_retention = retention.iloc[:, 0].mean()
        month_1_retention = retention.iloc[:, 1].mean() if retention.shape[1] > 1 else 0
        retention_drop = month_0_retention - month_1_retention
        insights["retention_drop_m0_to_m1"] = (
            f"{retention_drop:.1f}% drop from month 0 to 1 "
            f"({month_0_retention:.1f}% → {month_1_retention:.1f}%)"
        )

        # 3. CLV insights
        avg_clv = clv["total_revenue"].mean()
        top_clv = clv["total_revenue"].iloc[0]
        insights["avg_clv"] = f"${avg_clv:,.2f}"
        insights["top_customer_clv"] = f"${top_clv:,.2f}"

        # 4. Segment comparison
        best_segment = clv_seg["total_revenue"].idxmax()
        best_segment_rev = clv_seg.loc[best_segment, "total_revenue"]
        insights["best_segment"] = f"{best_segment} (${best_segment_rev:,.0f})"

        return insights

    # ──────────────────────────────────────────────────────────────────────
    # Full Analysis
    # ──────────────────────────────────────────────────────────────────────

    def run_full_analysis(self) -> dict:
        """Execute complete cohort analysis."""
        print("\n" + "=" * 60)
        print("  COHORT ANALYSIS")
        print("=" * 60)

        self.create_cohorts()

        print("\n👥 Creating Cohorts...")
        print(f"   Total unique customers: {self.cohorts['customer_id'].nunique():,}")
        print(f"   Date range: {self.cohorts['order_date'].min().date()} to {self.cohorts['order_date'].max().date()}")

        print("\n📊 1. Retention Rate by Cohort (%)")
        retention = self.retention_table()
        print(retention.head(10).to_string())

        print("\n💰 2. Revenue by Cohort")
        revenue = self.cohort_revenue()
        print(revenue.head(10).to_string())

        print("\n💵 3. Average Order Value by Cohort")
        aov = self.avg_order_value_by_cohort()
        print(aov.head(10).to_string())

        print("\n👤 4. Customer Lifetime Value")
        clv = self.customer_lifetime_value()
        print(clv.head(10).to_string())

        print("\n🏆 5. CLV by Customer Segment")
        clv_seg = self.clv_by_segment()
        print(clv_seg.to_string())

        print("\n" + "=" * 60)
        print("  COHORT INSIGHTS")
        print("=" * 60)
        insights = self.cohort_insights()
        for key, value in insights.items():
            print(f"\n   ✅ {key.replace('_', ' ').title()}: {value}")

        results = {
            "retention_table": retention,
            "revenue_table": revenue,
            "aov_table": aov,
            "customer_lifetime_value": clv,
            "clv_by_segment": clv_seg,
            "insights": insights,
        }

        return results
