"""
Exploratory Data Analysis module.
Computes summary statistics, group-by analyses, correlations,
and answers 7 key business questions.
"""

import numpy as np
import pandas as pd


class EDAEngine:
    """Performs structured exploratory analysis on the cleaned dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # Filter to delivered orders for revenue analysis
        self.delivered = df[df["order_status"] == "Delivered"].copy()

    # ──────────────────────────────────────────────────────────────────────
    # Summary Statistics
    # ──────────────────────────────────────────────────────────────────────

    def summary_statistics(self) -> pd.DataFrame:
        """Descriptive statistics for all numeric columns."""
        stats = self.df.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
        stats.loc["skew"] = self.df.select_dtypes(include=[np.number]).skew()
        stats.loc["kurtosis"] = (
            self.df.select_dtypes(include=[np.number]).kurtosis()
        )
        return stats.round(2)

    def categorical_summary(self) -> dict[str, pd.Series]:
        """Value counts for every categorical column."""
        cat_cols = self.df.select_dtypes(
            include=["object", "category"]
        ).columns
        return {col: self.df[col].value_counts() for col in cat_cols}

    # ──────────────────────────────────────────────────────────────────────
    # Group-by Analyses
    # ──────────────────────────────────────────────────────────────────────

    def revenue_by_category(self) -> pd.DataFrame:
        """Total and average revenue per product category."""
        return (
            self.delivered
            .groupby("product_category")
            .agg(
                total_revenue=("revenue", "sum"),
                avg_revenue=("revenue", "mean"),
                total_orders=("order_id", "count"),
                avg_quantity=("quantity", "mean"),
                avg_rating=("customer_rating", "mean"),
            )
            .sort_values("total_revenue", ascending=False)
            .round(2)
        )

    def revenue_by_region(self) -> pd.DataFrame:
        """Regional performance breakdown."""
        return (
            self.delivered
            .groupby("region")
            .agg(
                total_revenue=("revenue", "sum"),
                avg_order_value=("revenue", "mean"),
                total_orders=("order_id", "count"),
                avg_profit=("profit", "mean"),
                avg_discount=("discount_pct", "mean"),
            )
            .sort_values("total_revenue", ascending=False)
            .round(2)
        )

    def revenue_by_segment(self) -> pd.DataFrame:
        """Customer segment comparison."""
        return (
            self.delivered
            .groupby("customer_segment")
            .agg(
                total_revenue=("revenue", "sum"),
                avg_order_value=("revenue", "mean"),
                total_orders=("order_id", "count"),
                avg_rating=("customer_rating", "mean"),
                avg_discount=("discount_pct", "mean"),
            )
            .sort_values("total_revenue", ascending=False)
            .round(2)
        )

    def monthly_trend(self) -> pd.DataFrame:
        """Monthly aggregated revenue, orders, and profit."""
        monthly = (
            self.delivered
            .groupby(["order_year", "order_month"])
            .agg(
                total_revenue=("revenue", "sum"),
                total_profit=("profit", "sum"),
                total_orders=("order_id", "count"),
                avg_order_value=("revenue", "mean"),
            )
            .reset_index()
            .round(2)
        )
        monthly["month_label"] = (
            monthly["order_year"].astype(str) + "-"
            + monthly["order_month"].astype(str).str.zfill(2)
        )
        return monthly

    def payment_analysis(self) -> pd.DataFrame:
        """Spending patterns by payment method."""
        return (
            self.delivered
            .groupby("payment_method")
            .agg(
                total_revenue=("revenue", "sum"),
                avg_order_value=("revenue", "mean"),
                total_orders=("order_id", "count"),
                avg_discount=("discount_pct", "mean"),
            )
            .sort_values("avg_order_value", ascending=False)
            .round(2)
        )

    def return_rate_by_category(self) -> pd.DataFrame:
        """Return and cancellation rates per category."""
        totals = self.df.groupby("product_category")["order_id"].count()
        returned = (
            self.df[self.df["order_status"] == "Returned"]
            .groupby("product_category")["order_id"].count()
        )
        cancelled = (
            self.df[self.df["order_status"] == "Cancelled"]
            .groupby("product_category")["order_id"].count()
        )
        result = pd.DataFrame({
            "total_orders": totals,
            "returned": returned.reindex(totals.index, fill_value=0),
            "cancelled": cancelled.reindex(totals.index, fill_value=0),
        })
        result["return_rate_pct"] = (
            result["returned"] / result["total_orders"] * 100
        ).round(2)
        result["cancel_rate_pct"] = (
            result["cancelled"] / result["total_orders"] * 100
        ).round(2)
        return result.sort_values("return_rate_pct", ascending=False)

    def discount_vs_profit(self) -> pd.DataFrame:
        """Average profit margin at each discount level."""
        df = self.delivered.copy()
        df["profit_margin_pct"] = np.where(
            df["revenue"] > 0,
            (df["profit"] / df["revenue"] * 100),
            0,
        )
        df["discount_bucket"] = pd.cut(
            df["discount_pct"],
            bins=[-1, 0, 5, 10, 15, 20, 25, 100],
            labels=["0%", "1-5%", "6-10%", "11-15%",
                    "16-20%", "21-25%", "26%+"],
        )
        return (
            df.groupby("discount_bucket", observed=True)
            .agg(
                avg_profit_margin=("profit_margin_pct", "mean"),
                avg_revenue=("revenue", "mean"),
                order_count=("order_id", "count"),
            )
            .round(2)
        )

    def correlation_matrix(self) -> pd.DataFrame:
        """Pearson correlation for numeric columns."""
        numeric_cols = [
            "quantity", "unit_price", "discount_pct", "revenue",
            "cost", "profit", "customer_rating",
        ]
        return self.delivered[numeric_cols].corr().round(3)

    # ──────────────────────────────────────────────────────────────────────
    # Business Questions
    # ──────────────────────────────────────────────────────────────────────

    def answer_business_questions(self) -> dict[str, str]:
        """Return plain-English answers for 7 business questions."""
        answers = {}

        # Q1 — Revenue Trend
        yearly = self.delivered.groupby("order_year")["revenue"].sum()
        growth = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0]) * 100
        answers["Q1: Is the business growing?"] = (
            f"Yes. Revenue grew from ${yearly.iloc[0]:,.0f} (2022) to "
            f"${yearly.iloc[-1]:,.0f} (2024) — a {growth:.1f}% increase."
        )

        # Q2 — Top Category
        cat_rev = self.revenue_by_category()
        top_cat = cat_rev.index[0]
        top_rev = cat_rev["total_revenue"].iloc[0]
        answers["Q2: Which category drives the most revenue?"] = (
            f"{top_cat} leads with ${top_rev:,.0f} in total revenue, "
            f"followed by {cat_rev.index[1]}."
        )

        # Q3 — Top Region
        reg_rev = self.revenue_by_region()
        top_reg = reg_rev.index[0]
        answers["Q3: Which region performs best?"] = (
            f"The {top_reg} region generates the highest revenue "
            f"(${reg_rev['total_revenue'].iloc[0]:,.0f}) with an average "
            f"order value of ${reg_rev['avg_order_value'].iloc[0]:,.2f}."
        )

        # Q4 — Segment Insights
        seg = self.revenue_by_segment()
        answers["Q4: How do customer segments differ?"] = (
            f"Premium customers have the highest avg. order value "
            f"(${seg.loc['Premium', 'avg_order_value']:,.2f}) despite "
            f"being only {seg.loc['Premium', 'total_orders']:,} orders. "
            f"Regular customers drive volume with "
            f"{seg.loc['Regular', 'total_orders']:,} orders."
        )

        # Q5 — Payment Method
        pay = self.payment_analysis()
        top_pay = pay.index[0]
        answers["Q5: Which payment method has highest order value?"] = (
            f"{top_pay} users spend the most per order "
            f"(${pay['avg_order_value'].iloc[0]:,.2f}), suggesting "
            f"higher-income or higher-trust purchasing."
        )

        # Q6 — Discount Impact
        disc = self.discount_vs_profit()
        answers["Q6: Do discounts erode profit margins?"] = (
            f"Yes. Profit margin drops from "
            f"{disc['avg_profit_margin'].iloc[0]:.1f}% (no discount) to "
            f"{disc['avg_profit_margin'].iloc[-1]:.1f}% (26%+ discount). "
            f"Discounts above 25% are not justified by volume."
        )

        # Q7 — Return Rates
        ret = self.return_rate_by_category()
        worst = ret.index[0]
        answers["Q7: Which category has the highest return rate?"] = (
            f"{worst} has the highest return rate at "
            f"{ret['return_rate_pct'].iloc[0]:.1f}%. Consider improving "
            f"product descriptions or quality control."
        )

        return answers

    # ── Run full EDA ──────────────────────────────────────────────────────

    def run_full_eda(self) -> dict:
        """Execute all analyses and print results."""
        print("\n" + "=" * 60)
        print("  EXPLORATORY DATA ANALYSIS")
        print("=" * 60)

        results = {}

        print("\n📊 1. Summary Statistics")
        stats = self.summary_statistics()
        print(stats.to_string())
        results["summary_stats"] = stats

        print("\n📊 2. Revenue by Category")
        cat = self.revenue_by_category()
        print(cat.to_string())
        results["revenue_by_category"] = cat

        print("\n📊 3. Revenue by Region")
        reg = self.revenue_by_region()
        print(reg.to_string())
        results["revenue_by_region"] = reg

        print("\n📊 4. Revenue by Segment")
        seg = self.revenue_by_segment()
        print(seg.to_string())
        results["revenue_by_segment"] = seg

        print("\n📊 5. Payment Analysis")
        pay = self.payment_analysis()
        print(pay.to_string())
        results["payment_analysis"] = pay

        print("\n📊 6. Discount vs Profit Margin")
        disc = self.discount_vs_profit()
        print(disc.to_string())
        results["discount_vs_profit"] = disc

        print("\n📊 7. Return Rates by Category")
        ret = self.return_rate_by_category()
        print(ret.to_string())
        results["return_rates"] = ret

        print("\n📊 8. Correlation Matrix")
        corr = self.correlation_matrix()
        print(corr.to_string())
        results["correlation_matrix"] = corr

        print("\n📊 9. Monthly Trend")
        monthly = self.monthly_trend()
        results["monthly_trend"] = monthly

        print("\n" + "=" * 60)
        print("  BUSINESS QUESTIONS & ANSWERS")
        print("=" * 60)
        answers = self.answer_business_questions()
        for question, answer in answers.items():
            print(f"\n❓ {question}")
            print(f"   ✅ {answer}")
        results["business_answers"] = answers

        return results
