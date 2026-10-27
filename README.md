Sales Data Analyzer & Visualization

A menu-driven Python project for CSV sales analysis. It demonstrates Pandas cleaning and DataFrame operations, NumPy array operations and statistics, filtering and sorting, grouping, pivot tables, CSV combining/splitting, and Matplotlib/Seaborn visualizations. An original synthetic dataset is included so it runs immediately offline.

## Setup and run (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src\sales_analyzer.py
```

If activation is blocked, run `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe src\sales_analyzer.py`.

macOS/Linux: `python -m venv .venv`, activate with `source .venv/bin/activate`, install requirements, then run `python src/sales_analyzer.py`.

## Interactive menu

1. Load included or user-supplied CSV; required columns are Date, Product, Region, Sales.
2. Explore head/tail, column names, data types, DataFrame info, and missing-value counts.
3. Perform grouped DataFrame operations, vectorized NumPy profit-margin calculation, and a region/category pivot table.
4. Show, fill, drop, or replace missing values.
5. Print descriptive statistics, variance, and grouped sales aggregates.
6. Generate bar, line, scatter, pie, histogram, stacked bar, or Seaborn heatmap charts.
7. Save current data as CSV.
8. Search text, filter region/minimum sales, and sort results.
9. Combine another compatible CSV.
10. Split the data into one CSV per region.
11. Demonstrate NumPy array conversion, indexing, slicing, arithmetic, splitting, concatenation, and statistics.
12. Exit.

Charts are saved under `figures/`; region splits are saved under `reports/regions/`.

## Included data and assumptions

`data/sales_data.csv` contains 24 **synthetic example orders**, not real business information. It is provided only to make the application immediately runnable. Replace it with a properly licensed dataset for real analysis and cite its source.

The loader requires `Date`, `Product`, `Region`, and `Sales`. The example additionally contains OrderID, Category, Units, UnitPrice, and Profit. The cleaning operation parses dates, removes duplicate rows, fills missing numeric fields with mean or median and missing text fields with `Unknown`. When Units and UnitPrice exist, it recalculates Sales as Units × UnitPrice; this is an explicit example assumption and should be adapted if sales is defined differently in your source data.
