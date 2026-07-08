# 🛒 E-Commerce Sales Performance Analysis (2022–2024)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview

A comprehensive **end-to-end data analysis** of a fictional e-commerce company's
sales data spanning **January 2022 to December 2024**. The dataset covers
**50,000+ transactions** across 4 regions, 6 product categories, and 3 customer
segments.

This project demonstrates a full analytics workflow — from raw data cleaning
through exploratory analysis to actionable business recommendations.

---

## 🎯 Business Questions Answered

| # | Question |
|---|----------|
| 1 | What is the overall revenue trend — is the business growing? |
| 2 | Which product categories and regions drive the most revenue? |
| 3 | How do customer segments differ in spending behavior? |
| 4 | What seasonal patterns exist and how should marketing respond? |
| 5 | Which payment methods are preferred and does it affect order value? |
| 6 | What is the relationship between discount levels and profit margin? |
| 7 | Which products have the highest return rates? |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas** — data manipulation and cleaning
- **NumPy** — numerical operations
- **Matplotlib + Seaborn** — data visualization
- **Jupyter Notebook** — interactive analysis

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/shabib liaqat/ecommerce-sales-analysis.git
cd ecommerce-sales-analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Execute the complete analysis pipeline
python main.py

# OR run individual steps:
python src/data_generator.py    # Generate synthetic dataset
python src/data_cleaning.py     # Clean the raw data
# Then run Jupyter for interactive analysis
jupyter notebook notebooks/ecommerce_analysis.ipynb
```

---

## 📊 Key Findings

1. **Revenue grew 34% from 2022 to 2024**, with strongest growth in Q4.
2. **Electronics & Clothing** account for 55% of total revenue.
3. **West & East regions** outperform South & Central by ~20% in average order value.
4. **Premium customers** represent only 18% of orders but contribute 35% of revenue.
5. **Discounts above 25%** erode profit margin without proportional volume increase.
6. **Credit Card & PayPal** users spend 15% more per order than Cash/Debit users.
7. **June–August and November–December** are peak seasons across all categories.

---

## 📁 Project Structure

```
ecommerce-sales-analysis/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── main.py                            # Main orchestrator script
├── .gitignore
│
├── data/
│   ├── ecommerce_sales_raw.csv       # Generated raw data
│   └── ecommerce_sales_clean.csv     # Cleaned data
│
├── notebooks/
│   └── ecommerce_analysis.ipynb      # Interactive Jupyter notebook
│
├── src/
│   ├── __init__.py
│   ├── data_generator.py             # Dataset generation
│   ├── data_cleaning.py              # Data cleaning pipeline
│   ├── eda.py                        # Exploratory analysis
│   └── visualizations.py             # Chart generation
│
├── reports/
│   └── business_insights.md          # Auto-generated insights
│
└── outputs/
    ├── plots/                        # Generated visualizations
    │   ├── 01_monthly_revenue_trend.png
    │   ├── 02_revenue_by_category.png
    │   ├── 03_region_performance.png
    │   ├── 04_correlation_heatmap.png
    │   ├── 05_customer_segment_analysis.png
    │   ├── 06_payment_method_distribution.png
    │   ├── 07_top_products_by_quantity.png
    │   └── 08_monthly_orders_by_region.png
    └── summary_statistics.csv
```

---

## 🔄 Data Pipeline

### 1. **Data Generation** (`data_generator.py`)
- Generates 50,000+ realistic transactions
- Injects intentional data quality issues:
  - Missing values (~3% customer ratings, 2% discounts, 1.5% regions)
  - Duplicate rows (~1%)
  - Inconsistent categorical values (mixed casing)
  - Negative revenue for returns/cancellations

### 2. **Data Cleaning** (`data_cleaning.py`)
- Removes exact duplicates
- Fixes data types (dates, numerics)
- Standardizes categorical values
- Handles missing values per column strategy:
  - `customer_rating` → median imputation
  - `discount_pct` → fill with 0
  - `region` / `shipping_type` → mode imputation
- Zeroes out negative revenue on returned orders
- Adds derived time columns (year, month, quarter, day_of_week)
- Validates final schema

### 3. **Exploratory Analysis** (`eda.py`)
- Summary statistics and distributions
- Revenue analysis by category, region, customer segment
- Monthly trends and seasonality patterns
- Payment method analysis
- Return rate investigations
- Discount impact on profitability
- Correlation analysis
- Answers 7 key business questions

### 4. **Visualizations** (`visualizations.py`)
- 8 publication-quality charts:
  1. Monthly revenue trend (bar + MA3 line)
  2. Revenue by category (dual horizontal bars)
  3. Regional performance (3-panel comparison)
  4. Correlation heatmap
  5. Customer segment analysis (pie + bars)
  6. Payment method distribution (donut + bars)
  7. Top 15 products by units sold
  8. Monthly orders heatmap by region

---

## 💼 Key Insights & Recommendations

### Revenue Growth
- **Finding**: 34% revenue growth (2022→2024)
- **Action**: Continue current strategic direction; scale successful initiatives

### Category Performance
- **Finding**: Electronics & Clothing = 55% of revenue
- **Action**: Expand SKU depth; invest in inventory for these categories

### Regional Strategy
- **Finding**: West & East regions have 20% higher AOV than South & Central
- **Action**: Allocate more marketing budget to high-AOV regions; investigate South/Central opportunities

### Customer Segmentation
- **Finding**: Premium segment: 18% of orders, 35% of revenue
- **Action**: Launch loyalty program; convert Regular → Premium customers

### Discount Strategy
- **Finding**: Discounts >25% destroy profit margins (-36% margin drop)
- **Action**: Implement 20% discount cap policy; use targeted, strategic discounts only

### Seasonality
- **Finding**: Peak months: June-Aug (90-120% of avg), Nov-Dec (140% of avg)
- **Action**: Pre-position inventory; increase marketing spend 1-2 months prior

### Quality & Returns
- **Finding**: High-return categories identified
- **Action**: Improve product descriptions, sizing guides, quality control

---

## 📊 Output Files

After running the pipeline, outputs are generated:

```
✅ data/ecommerce_sales_raw.csv        (50,100 rows × 18 cols)
✅ data/ecommerce_sales_clean.csv      (49,650 rows × 23 cols)
✅ outputs/plots/01-08_*.png           (8 visualizations)
✅ outputs/summary_statistics.csv      (Key metrics summary)
✅ reports/business_insights.md        (Findings & recommendations)
```

---

## 🔧 Customization

### Modify Dataset Parameters
Edit `src/data_generator.py`:
```python
NUM_ROWS = 50_000           # Dataset size
REGIONS = ["West", "East", "South", "Central"]
CATEGORIES = ["Electronics", "Clothing", ...]
PRICE_RANGES = {...}        # Category price bands
```

### Adjust Cleaning Logic
Edit `src/data_cleaning.py`:
```python
# Change imputation strategies
df["customer_rating"].fillna(median_rating)  # or use mean, specific value
df["region"].fillna(mode_region)             # or drop rows
```

### Add New Analyses
Edit `src/eda.py` and `src/visualizations.py`:
```python
def new_analysis(self) -> pd.DataFrame:
    """Your custom analysis here."""
    return results

def plot_new_chart(self):
    """Your custom visualization here."""
    # ... plotting code
```

---

## 📈 Performance Notes

- **Data generation**: ~2-3 seconds (50K rows)
- **Data cleaning**: ~1 second
- **EDA**: ~3-5 seconds (8 analyses)
- **Visualizations**: ~10-15 seconds (8 charts, 150 DPI)
- **Total pipeline**: ~20-30 seconds

---

## 📜 License

This project is licensed under the MIT License.

---

## 👤 Author

**Your Name**
- LinkedIn: [shabibliaqat](https://linkedin.com/in/shabibliaqat)
- GitHub: [shabibliaqa4-svg](https://github.com/shabibliaqa4-svg)

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Changelog

### v1.0.0 (2024-01-15)
- Initial project setup
- Complete data pipeline (generation, cleaning, EDA)
- 8 publication-quality visualizations
- Business insights report generation
- Jupyter notebook for interactive analysis

---

## ❓ FAQ

**Q: Can I use real data instead of generated data?**
A: Yes! Replace `data/ecommerce_sales_raw.csv` with your own data. Ensure it has the same column names and types.

**Q: How do I add more visualizations?**
A: Add a new method to the `DashboardVisualizer` class in `src/visualizations.py`, then call it in `generate_all()`.

**Q: Can I modify the business questions?**
A: Yes! Edit the `answer_business_questions()` method in `src/eda.py`.

**Q: How do I adapt this for a different business domain?**
A: Update column names, categories, and business metrics across generator, cleaner, and analysis modules.

---
# Ecommerece-sales-analysis
