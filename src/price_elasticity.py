"""
Price Elasticity Analysis module - pricing optimization and revenue maximization.
"""

import numpy as np
import pandas as pd
from scipy import stats


class PriceElasticityAnalyzer:
    """Analyzes price sensitivity and optimal pricing strategies."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["order_date"] = pd.to_datetime(self.df["order_date"])

    # ──────────────────────────────────────────────────────────────────────
    # Price Elasticity Calculation
    # ──────────────────────────────────────────────────────────────────────

    def calculate_elasticity_by_category(self) -> pd.DataFrame:
        """
        Calculate price elasticity of demand (PED) by category.
        PED = % change in quantity / % change in price
        PED < -1 = elastic (sensitive to price changes)
        PED > -1 = inelastic (insensitive to price changes)
        """
        delivered = self.df[self.df["order_status"] == "Delivered"].copy()

        results = []

        for category in delivered["product_category"].unique():
            cat_data = delivered[delivered["product_category"] == category].copy()

            if len(cat_data) < 10:
                continue

            # Create price bins to analyze demand at different price points
            cat_data["price_bin"] = pd.qcut(
                cat_data["unit_price"],
                q=5,
                duplicates="drop",
                labels=False
            )

            # Calculate quantity and price for each bin
            elasticity_data = (
                cat_data
                .groupby("price_bin")
                .agg(
                    avg_price=("unit_price", "mean"),
                    total_quantity=("quantity", "sum"),
                    avg_quantity=("quantity", "mean"),
                    order_count=("order_id", "count"),
                )
                .reset_index()
            )

            if len(elasticity_data) < 2:
                continue

            # Calculate PED using regression
            log_prices = np.log(elasticity_data["avg_price"])
            log_quantities = np.log(elasticity_data["avg_quantity"])

            # Remove any inf or nan
            valid_idx = np.isfinite(log_prices) & np.isfinite(log_quantities)
            if valid_idx.sum() < 2:
                continue

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_prices[valid_idx],
                log_quantities[valid_idx]
            )

            # Negative slope indicates normal demand curve
            ped = slope if not np.isnan(slope) else 0
            elasticity = "Elastic" if ped < -1 else "Inelastic"

            results.append({
                "category": category,
                "price_elasticity": round(ped, 2),
                "elasticity_type": elasticity,
                "r_squared": round(r_value ** 2, 3),
                "min_price": round(cat_data["unit_price"].min(), 2),
                "max_price": round(cat_data["unit_price"].max(), 2),
                "avg_price": round(cat_data["unit_price"].mean(), 2),
                "avg_quantity_per_order": round(cat_data["quantity"].mean(), 2),
            })

        return pd.DataFrame(results).sort_values("price_elasticity")

    # ──────────────────────────────────────────────────────────────────────
    # Price-Revenue Optimization
    # ──────────────────────────────────────────────────────────────────────

    def revenue_at_price_points(self) -> pd.DataFrame:
        """Simulate revenue at different price points by category."""
        delivered = self.df[self.df["order_status"] == "Delivered"].copy()

        results = []

        for category in delivered["product_category"].unique():
            cat_data = delivered[delivered["product_category"] == category].copy()

            if len(cat_data) < 10:
                continue

            current_revenue = cat_data["revenue"].sum()
            current_avg_price = cat_data["unit_price"].mean()
            current_qty = cat_data["quantity"].sum()

            # Get elasticity
            elasticity_df = self.calculate_elasticity_by_category()
            cat_elasticity = elasticity_df[
                elasticity_df["category"] == category
            ]["price_elasticity"].values

            if len(cat_elasticity) == 0:
                ped = -1.0
            else:
                ped = cat_elasticity[0]

            # Simulate price scenarios (-20% to +20%)
            price_scenarios = [0.80, 0.90, 1.0, 1.10, 1.20]

            for scenario in price_scenarios:
                new_price = current_avg_price * scenario
                # Calculate expected quantity change using elasticity
                price_change_pct = scenario - 1
                qty_change_pct = ped * price_change_pct
                new_qty = current_qty * (1 + qty_change_pct)

                new_revenue = new_price * new_qty
                revenue_change = new_revenue - current_revenue
                revenue_change_pct = (revenue_change / current_revenue * 100)

                results.append({
                    "category": category,
                    "price_scenario": f"{scenario*100:.0f}%",
                    "current_price": round(current_avg_price, 2),
                    "simulated_price": round(new_price, 2),
                    "expected_revenue": round(new_revenue, 0),
                    "current_revenue": round(current_revenue, 0),
                    "revenue_change": round(revenue_change, 0),
                    "revenue_change_pct": round(revenue_change_pct, 1),
                })

        return pd.DataFrame(results)

    # ──────────────────────────────────────────────────────────────────────
    # Optimal Pricing
    # ──────────────────────────────────────────────────────────────────────

    def optimal_pricing_recommendation(self) -> pd.DataFrame:
        """Recommend optimal prices based on elasticity and profit margins."""
        elasticity = self.calculate_elasticity_by_category()
        revenue_sim = self.revenue_at_price_points()

        recommendations = []

        for category in elasticity["category"].unique():
            cat_elasticity = elasticity[elasticity["category"] == category].iloc[0]
            cat_revenue_sims = revenue_sim[revenue_sim["category"] == category]

            # Find price point that maximizes revenue
            best_scenario = cat_revenue_sims.loc[
                cat_revenue_sims["revenue_change_pct"].idxmax()
            ]

            ped = cat_elasticity["price_elasticity"]

            # Strategy recommendation based on elasticity
            if ped < -1.5:
                strategy = "⬇️ Lower prices - High elasticity, price cuts increase revenue"
            elif ped < -1:
                strategy = "📊 Consider 5-10% price decrease - Moderate elasticity"
            elif ped < -0.5:
                strategy = "💰 Keep current price - Low elasticity, demand insensitive"
            else:
                strategy = "⬆️ Raise prices - Very low elasticity, high margin potential"

            recommendations.append({
                "category": category,
                "current_avg_price": cat_elasticity["avg_price"],
                "price_elasticity": cat_elasticity["price_elasticity"],
                "optimal_price_scenario": best_scenario["price_scenario"],
                "optimal_price": best_scenario["simulated_price"],
                "potential_revenue_increase": f"{best_scenario['revenue_change_pct']:.1f}%",
                "expected_new_revenue": int(best_scenario["expected_revenue"]),
                "strategy": strategy,
            })

        return pd.DataFrame(recommendations).sort_values(
            "potential_revenue_increase",
            ascending=False
        )

    # ──────────────────────────────────────────────────────────────────────
    # Price vs Profit Analysis
    # ──────────────────────────────────────────────────────────────────────

    def margin_by_price_tier(self) -> pd.DataFrame:
        """Analyze profit margins across price tiers."""
        delivered = self.df[self.df["order_status"] == "Delivered"].copy()

        results = []

        for category in delivered["product_category"].unique():
            cat_data = delivered[delivered["product_category"] == category].copy()

            if len(cat_data) < 10:
                continue

            # Create price tiers
            cat_data["price_tier"] = pd.qcut(
                cat_data["unit_price"],
                q=3,
                labels=["Budget", "Mid-Range", "Premium"],
                duplicates="drop"
            )

            tier_analysis = (
                cat_data
                .groupby("price_tier")
                .agg(
                    avg_price=("unit_price", "mean"),
                    avg_revenue=("revenue", "mean"),
                    avg_profit=("profit", "mean"),
                    profit_margin_pct=(
                        "profit",
                        lambda x: (x.sum() / cat_data.loc[x.index, "revenue"].sum() * 100) if cat_data.loc[x.index, "revenue"].sum() > 0 else 0
                    ),
                    order_count=("order_id", "count"),
                    avg_discount=("discount_pct", "mean"),
                )
                .reset_index()
            )

            tier_analysis["category"] = category
            results.append(tier_analysis)

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    # ──────────────────────────────────────────────────────────────────────
    # Full Analysis
    # ──────────────────────────────────────────────────────────────────────

    def run_full_analysis(self) -> dict:
        """Execute complete price elasticity analysis."""
        print("\n" + "=" * 60)
        print("  PRICE ELASTICITY & OPTIMIZATION ANALYSIS")
        print("=" * 60)

        print("\n📊 1. Price Elasticity of Demand by Category")
        elasticity = self.calculate_elasticity_by_category()
        print(elasticity.to_string(index=False))

        print("\n💹 2. Revenue Simulation at Different Price Points")
        revenue_sim = self.revenue_at_price_points()
        print(revenue_sim.groupby("category")[
            ["category", "price_scenario", "revenue_change_pct"]
        ].head(10).to_string(index=False))

        print("\n🎯 3. Optimal Pricing Recommendations")
        optimal = self.optimal_pricing_recommendation()
        print(optimal.to_string(index=False))

        print("\n💰 4. Profit Margin by Price Tier")
        margins = self.margin_by_price_tier()
        if not margins.empty:
            print(margins.to_string(index=False))

        print("\n" + "=" * 60)
        print("  STRATEGIC PRICING INSIGHTS")
        print("=" * 60)
        print(optimal[["category", "strategy"]].to_string(index=False))

        results = {
            "elasticity": elasticity,
            "revenue_simulation": revenue_sim,
            "optimal_pricing": optimal,
            "margin_by_tier": margins,
        }

        return results
