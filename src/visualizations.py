"""
Visualization module — generates 8 publication-quality charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

PLOT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "plots",
)
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Global style ─────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = sns.color_palette("Set2", 10)
COLOR_PRIMARY = "#2E86AB"
COLOR_SECONDARY = "#A23B72"
COLOR_ACCENT = "#F18F01"
COLOR_POSITIVE = "#2CA58D"
COLOR_NEGATIVE = "#E63946"


def _save(fig: plt.Figure, filename: str):
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  📈 Saved → {path}")


class DashboardVisualizer:
    """Creates all project visualizations from EDA results."""

    def __init__(self, df: pd.DataFrame, eda_results: dict):
        self.df = df.copy()
        self.delivered = df[df["order_status"] == "Delivered"].copy()
        self.results = eda_results

    # ── Plot 1: Monthly Revenue Trend ─────────────────────────────────────

    def plot_monthly_revenue_trend(self):
        monthly = self.results["monthly_trend"]

        fig, ax1 = plt.subplots(figsize=(16, 7))

        # Revenue bars
        bars = ax1.bar(
            monthly["month_label"],
            monthly["total_revenue"],
            color=COLOR_PRIMARY,
            alpha=0.7,
            label="Monthly Revenue",
            width=0.7,
        )

        # 3-month moving average line
        monthly["revenue_ma3"] = (
            monthly["total_revenue"].rolling(window=3, min_periods=1).mean()
        )
        ax1.plot(
            monthly["month_label"],
            monthly["revenue_ma3"],
            color=COLOR_ACCENT,
            linewidth=2.5,
            marker="o",
            markersize=4,
            label="3-Month Moving Avg",
        )

        ax1.set_xlabel("Month", fontsize=12)
        ax1.set_ylabel("Revenue ($)", fontsize=12)
        ax1.set_title(
            "Monthly Revenue Trend (2022–2024)",
            fontsize=16, fontweight="bold", pad=15,
        )
        ax1.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")
        )
        ax1.tick_params(axis="x", rotation=90, labelsize=8)
        ax1.legend(fontsize=11, loc="upper left")
        ax1.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        _save(fig, "01_monthly_revenue_trend.png")

    # ── Plot 2: Revenue by Category (Horizontal Bar) ─────────────────────

    def plot_revenue_by_category(self):
        cat = self.results["revenue_by_category"].sort_values("total_revenue")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Total revenue
        bars = axes[0].barh(
            cat.index, cat["total_revenue"],
            color=sns.color_palette("Blues_d", len(cat)),
            edgecolor="white",
        )
        axes[0].set_xlabel("Total Revenue ($)")
        axes[0].set_title("Total Revenue by Category", fontweight="bold")
        axes[0].xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")
        )
        for bar, val in zip(bars, cat["total_revenue"]):
            axes[0].text(
                val + cat["total_revenue"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}",
                va="center", fontsize=9,
            )

        # Right: Average order value
        axes[1].barh(
            cat.index, cat["avg_revenue"],
            color=sns.color_palette("Oranges_d", len(cat)),
            edgecolor="white",
        )
        axes[1].set_xlabel("Avg Order Value ($)")
        axes[1].set_title("Avg Order Value by Category", fontweight="bold")

        fig.suptitle(
            "Product Category Performance",
            fontsize=16, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        _save(fig, "02_revenue_by_category.png")

    # ── Plot 3: Region Performance ────────────────────────────────────────

    def plot_region_performance(self):
        reg = self.results["revenue_by_region"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_POSITIVE]

        # Total Revenue
        axes[0].bar(reg.index, reg["total_revenue"], color=colors)
        axes[0].set_title("Total Revenue", fontweight="bold")
        axes[0].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")
        )

        # Avg Order Value
        axes[1].bar(reg.index, reg["avg_order_value"], color=colors)
        axes[1].set_title("Avg Order Value", fontweight="bold")
        axes[1].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
        )

        # Order Volume
        axes[2].bar(reg.index, reg["total_orders"], color=colors)
        axes[2].set_title("Total Orders", fontweight="bold")
        axes[2].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e3:.1f}K")
        )

        fig.suptitle(
            "Regional Performance Comparison",
            fontsize=16, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        _save(fig, "03_region_performance.png")

    # ── Plot 4: Correlation Heatmap ───────────────────────────────────────

    def plot_correlation_heatmap(self):
        corr = self.results["correlation_matrix"]

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax,
            vmin=-1, vmax=1,
        )
        ax.set_title(
            "Feature Correlation Matrix",
            fontsize=16, fontweight="bold", pad=15,
        )
        fig.tight_layout()
        _save(fig, "04_correlation_heatmap.png")

    # ── Plot 5: Customer Segment Analysis ─────────────────────────────────

    def plot_customer_segment_analysis(self):
        seg = self.results["revenue_by_segment"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        colors = [COLOR_POSITIVE, COLOR_PRIMARY, COLOR_ACCENT]

        # Revenue share (pie)
        axes[0].pie(
            seg["total_revenue"],
            labels=seg.index,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            textprops={"fontsize": 11},
        )
        axes[0].set_title("Revenue Share", fontweight="bold")

        # Avg Order Value (bar)
        axes[1].bar(seg.index, seg["avg_order_value"], color=colors)
        axes[1].set_title("Avg Order Value ($)", fontweight="bold")
        for i, (idx, val) in enumerate(
            seg["avg_order_value"].items()
        ):
            axes[1].text(
                i, val + 2, f"${val:,.0f}",
                ha="center", fontsize=10, fontweight="bold",
            )

        # Order Count (bar)
        axes[2].bar(seg.index, seg["total_orders"], color=colors)
        axes[2].set_title("Order Count", fontweight="bold")
        axes[2].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e3:.1f}K")
        )

        fig.suptitle(
            "Customer Segment Analysis",
            fontsize=16, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        _save(fig, "05_customer_segment_analysis.png")

    # ── Plot 6: Payment Method Distribution ───────────────────────────────

    def plot_payment_method_distribution(self):
        pay = self.results["payment_analysis"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Donut chart
        wedges, texts, autotexts = axes[0].pie(
            pay["total_orders"],
            labels=pay.index,
            autopct="%1.1f%%",
            colors=PALETTE[:len(pay)],
            startangle=90,
            pctdistance=0.85,
            textprops={"fontsize": 10},
        )
        centre_circle = plt.Circle((0, 0), 0.60, fc="white")
        axes[0].add_artist(centre_circle)
        axes[0].set_title("Order Distribution", fontweight="bold")

        # Avg order value
        bars = axes[1].barh(
            pay.index, pay["avg_order_value"],
            color=PALETTE[:len(pay)],
        )
        axes[1].set_xlabel("Avg Order Value ($)")
        axes[1].set_title("Avg Spend by Payment Method", fontweight="bold")
        for bar, val in zip(bars, pay["avg_order_value"]):
            axes[1].text(
                val + 1, bar.get_y() + bar.get_height() / 2,
                f"${val:,.2f}", va="center", fontsize=10,
            )

        fig.suptitle(
            "Payment Method Analysis",
            fontsize=16, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        _save(fig, "06_payment_method_distribution.png")

    # ── Plot 7: Top 15 Products by Quantity Sold ──────────────────────────

    def plot_top_products(self, n: int = 15):
        top = (
            self.delivered
            .groupby("product_name")["quantity"]
            .sum()
            .sort_values(ascending=True)
            .tail(n)
        )

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = sns.color_palette("viridis", len(top))
        ax.barh(top.index, top.values, color=colors)
        ax.set_xlabel("Total Units Sold", fontsize=12)
        ax.set_title(
            f"Top {n} Products by Units Sold",
            fontsize=16, fontweight="bold", pad=15,
        )
        for i, val in enumerate(top.values):
            ax.text(val + top.max() * 0.005, i,
                    f"{val:,}", va="center", fontsize=9)
        fig.tight_layout()
        _save(fig, "07_top_products_by_quantity.png")

    # ── Plot 8: Monthly Orders Heatmap by Region ─────────────────────────

    def plot_monthly_orders_heatmap(self):
        pivot = (
            self.delivered
            .groupby(["region", "order_month"])["order_id"]
            .count()
            .unstack(fill_value=0)
        )
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

        fig, ax = plt.subplots(figsize=(14, 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(
            "Monthly Order Volume by Region",
            fontsize=16, fontweight="bold", pad=15,
        )
        ax.set_ylabel("")
        fig.tight_layout()
        _save(fig, "08_monthly_orders_by_region.png")

    # ── Generate all plots ────────────────────────────────────────────────

    def generate_all(self):
        """Create and save every visualization."""
        print("\n" + "=" * 60)
        print("  GENERATING VISUALIZATIONS")
        print("=" * 60)

        self.plot_monthly_revenue_trend()
        self.plot_revenue_by_category()
        self.plot_region_performance()
        self.plot_correlation_heatmap()
        self.plot_customer_segment_analysis()
        self.plot_payment_method_distribution()
        self.plot_top_products()
        self.plot_monthly_orders_heatmap()

        print("=" * 60)
        print(f"  ✅ All 8 plots saved to {PLOT_DIR}")
        print("=" * 60 + "\n")
