"""Menu-driven sales CSV analyzer using Pandas, NumPy, Matplotlib, and Seaborn."""
from __future__ import annotations
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "sales_data.csv"
REQUIRED = {"Date", "Product", "Region", "Sales"}
NUMERIC = ["Units", "UnitPrice", "Sales", "Profit"]


class SalesDataAnalyzer:
    """Load and analyze a sales DataFrame, with reusable operations."""

    def __init__(self, file_path: str | Path | None = None):
        self.data = pd.DataFrame()
        self.file_path = None
        if file_path:
            self.load_data(file_path)

    def load_data(self, file_path: str | Path) -> None:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")
        frame = pd.read_csv(path)
        missing = REQUIRED - set(frame.columns)
        if missing:
            raise ValueError(f"CSV must include: {', '.join(sorted(missing))}")
        self.data, self.file_path = frame, path.resolve()
        print(f"Loaded {len(frame)} rows from {path}")

    def _ready(self) -> bool:
        if self.data.empty:
            print("Load a dataset first (menu option 1).")
            return False
        return True

    def explore_data(self) -> None:
        if not self._ready(): return
        print("\nFirst rows:\n", self.data.head().to_string(index=False))
        print("\nLast rows:\n", self.data.tail().to_string(index=False))
        print("\nColumns:", list(self.data.columns))
        print("\nTypes:\n", self.data.dtypes.to_string())
        print("\nDataFrame info:")
        self.data.info()
        print("\nMissing by column:\n", self.data.isna().sum().to_string())

    def clean_data(self, strategy="median") -> None:
        if not self._ready(): return
        before = len(self.data)
        self.data = self.data.drop_duplicates().copy()
        if "Date" in self.data:
            self.data["Date"] = pd.to_datetime(self.data["Date"], errors="coerce")
        for col in NUMERIC:
            if col in self.data:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
                if self.data[col].isna().any():
                    fill = self.data[col].mean() if strategy == "mean" else self.data[col].median()
                    self.data[col] = self.data[col].fillna(fill if pd.notna(fill) else 0)
        for col in self.data.select_dtypes(include="object"):
            self.data[col] = self.data[col].fillna("Unknown")
        if {"Units", "UnitPrice", "Sales"}.issubset(self.data.columns):
            self.data["Sales"] = self.data["Units"] * self.data["UnitPrice"]
        print(f"Removed {before-len(self.data)} duplicate rows; cleaned missing values/types.")

    def dataframe_operations(self) -> None:
        if not self._ready(): return
        print("\nRegional sales aggregate:\n", self.data.groupby("Region")["Sales"].agg(["sum", "mean", "count"]).to_string())
        if "Profit" in self.data:
            self.data["ProfitMarginPct"] = np.where(self.data.Sales != 0, self.data.Profit / self.data.Sales * 100, 0)
            print("Added ProfitMarginPct with vectorized NumPy math.")
        if "Category" in self.data:
            pivot = pd.pivot_table(self.data, index="Region", columns="Category", values="Sales", aggfunc="sum", fill_value=0)
            print("\nPivot table (region × category):\n", pivot.to_string())
        print("\nTop sales rows:\n", self.data.nlargest(min(5, len(self.data)), "Sales").to_string(index=False))

    def array_demo(self) -> None:
        if not self._ready(): return
        a = self.data["Sales"].to_numpy(dtype=float)
        left, right = np.array_split(a, 2)
        print("Array:", a, "\nFirst three (slice):", a[:3], "\nSales + 10:", a + 10,
              "\nSales × 2:", a * 2, "\nSplit:", left, "|", right,
              "\nConcatenated:", np.concatenate([left, right]))
        print(f"mean={np.mean(a):.2f}, std={np.std(a):.2f}, variance={np.var(a):.2f}, p90={np.percentile(a,90):.2f}")

    def statistics(self) -> None:
        if not self._ready(): return
        numeric = self.data.select_dtypes(include="number")
        print("\nDescriptive statistics:\n", numeric.describe(percentiles=[.25,.5,.75,.9]).to_string())
        print("\nVariance:\n", numeric.var().to_string())
        print("\nRegion/product totals:\n", self.data.groupby(["Region","Product"])["Sales"].agg(["sum","mean","count"]).to_string())

    def search_sort_filter(self) -> None:
        if not self._ready(): return
        term = input("Search text (blank skips): ").strip()
        result = self.data
        if term:
            mask = result.astype(str).apply(lambda c: c.str.contains(re.escape(term), case=False, na=False)).any(axis=1)
            result = result[mask]
        region = input("Region filter (blank for all): ").strip()
        if region:
            result = result[result.Region.astype(str).str.casefold() == region.casefold()]
        floor = input("Minimum sales (blank skips): ").strip()
        if floor:
            try: result = result[result.Sales >= float(floor)]
            except ValueError: print("Invalid number; skipped minimum filter.")
        column = input("Sort descending by column [Sales]: ").strip() or "Sales"
        if column not in result.columns:
            print("Column not found; sorting by Sales."); column = "Sales"
        print(f"\n{len(result)} matching rows:\n", result.sort_values(column, ascending=False).to_string(index=False))

    def handle_missing(self) -> None:
        if not self._ready(): return
        print("1 Show missing  2 Fill mean/median  3 Drop incomplete rows  4 Replace a column")
        choice = input("Choose: ").strip()
        if choice == "1":
            print(self.data[self.data.isna().any(axis=1)].to_string(index=False) or "No missing rows.")
        elif choice == "2":
            strategy = input("Fill numeric values with mean or median? [median]: ").strip().lower()
            self.clean_data("mean" if strategy == "mean" else "median")
        elif choice == "3":
            before = len(self.data); self.data = self.data.dropna().copy()
            print(f"Removed {before-len(self.data)} incomplete rows.")
        elif choice == "4":
            col = input(f"Column {list(self.data.columns)}: ").strip()
            if col not in self.data: print("Unknown column."); return
            value = input("Replacement: ")
            if pd.api.types.is_numeric_dtype(self.data[col]):
                try: value = float(value)
                except ValueError: print("Enter a number for this numeric column."); return
            count = int(self.data[col].isna().sum()); self.data[col] = self.data[col].fillna(value)
            print(f"Replaced {count} values.")
        else: print("Unknown choice.")

    def combine_csv(self, path: str) -> None:
        if not self._ready(): return
        try:
            other = pd.read_csv(Path(path).expanduser())
            if REQUIRED - set(other.columns): raise ValueError(f"Second CSV must include {sorted(REQUIRED)}")
            self.data = pd.concat([self.data, other], ignore_index=True, sort=False)
            print(f"Combined tables; now {len(self.data)} rows.")
        except (OSError, ValueError, pd.errors.ParserError) as exc: print(f"Could not combine CSV: {exc}")

    def split_by_region(self) -> None:
        if not self._ready(): return
        folder = ROOT / "reports" / "regions"; folder.mkdir(parents=True, exist_ok=True)
        for region, frame in self.data.groupby("Region", dropna=False):
            name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(region)).strip("_").lower() or "unknown"
            frame.to_csv(folder / f"{name}.csv", index=False)
            print(f"Saved reports/regions/{name}.csv ({len(frame)} rows)")

    def visualize(self, choice: str) -> None:
        if not self._ready(): return
        folder = ROOT / "figures"; folder.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", palette="deep")
        fig, ax = plt.subplots(figsize=(9, 5.5)); d = self.data
        if choice == "1":
            d.groupby("Region").Sales.sum().sort_values().plot.bar(ax=ax); ax.set(title="Total sales by region", ylabel="Sales")
        elif choice == "2":
            dated = d.assign(Date=pd.to_datetime(d.Date, errors="coerce")).dropna(subset=["Date"]).set_index("Date").sort_index()
            dated.Sales.resample("MS").sum().plot(marker="o", ax=ax); ax.set(title="Monthly sales trend", ylabel="Sales")
        elif choice == "3":
            sns.scatterplot(data=d, x="Sales", y="Profit" if "Profit" in d else "Sales", hue="Region", style="Category" if "Category" in d else None, ax=ax); ax.set(title="Sales and profit relationship")
        elif choice == "4":
            totals=d.groupby("Product").Sales.sum().sort_values(ascending=False); ax.pie(totals,labels=totals.index,autopct="%1.1f%%",startangle=90); ax.set(title="Sales share by product")
        elif choice == "5":
            sns.histplot(data=d,x="Sales",bins=min(12,max(5,len(d)//2)),kde=True,ax=ax); ax.set(title="Sales distribution")
        elif choice == "6":
            d.pivot_table(index="Region",columns="Category",values="Sales",aggfunc="sum",fill_value=0).plot(kind="bar",stacked=True,ax=ax); ax.set(title="Stacked sales by region/category",ylabel="Sales")
        elif choice == "7":
            p=d.pivot_table(index="Region",columns="Category",values="Sales",aggfunc="sum",fill_value=0); sns.heatmap(p,annot=True,fmt=".0f",cmap="Blues",ax=ax); ax.set(title="Sales heatmap")
        else: plt.close(fig); print("Unknown chart."); return
        if choice != "4": ax.tick_params(axis="x",rotation=25)
        fig.tight_layout(); name=["bar","line","scatter","pie","histogram","stacked","heatmap"][int(choice)-1]
        path=folder/f"{name}.png"; fig.savefig(path,dpi=170,bbox_inches="tight"); plt.close(fig)
        print(f"Saved chart to {path}")


def menu() -> None:
    app = SalesDataAnalyzer(DEFAULT_CSV)
    options = "\n=== SALES DATA ANALYZER ===\n1 Load CSV\n2 Explore data\n3 DataFrame operations and pivot\n4 Handle missing values\n5 Descriptive statistics\n6 Visualize\n7 Save current data\n8 Search, sort, filter\n9 Combine another CSV\n10 Split by region\n11 NumPy operations\n12 Exit"
    while True:
        print(options); choice=input("Select: ").strip()
        if choice == "1":
            path=input(f"CSV path [{DEFAULT_CSV}]: ").strip() or str(DEFAULT_CSV)
            try: app.load_data(path)
            except (OSError, ValueError, pd.errors.ParserError) as exc: print(f"Could not load CSV: {exc}")
        elif choice == "2": app.explore_data()
        elif choice == "3": app.dataframe_operations()
        elif choice == "4": app.handle_missing()
        elif choice == "5": app.statistics()
        elif choice == "6": app.visualize(input("1 Bar 2 Line 3 Scatter 4 Pie 5 Histogram 6 Stacked 7 Seaborn heatmap: ").strip())
        elif choice == "7" and app._ready():
            p=Path(input("Output CSV [data/cleaned_sales.csv]: ").strip() or "data/cleaned_sales.csv")
            if not p.is_absolute(): p=ROOT/p
            p.parent.mkdir(parents=True,exist_ok=True); app.data.to_csv(p,index=False); print(f"Saved {p}")
        elif choice == "8": app.search_sort_filter()
        elif choice == "9": app.combine_csv(input("Second CSV path: ").strip())
        elif choice == "10": app.split_by_region()
        elif choice == "11": app.array_demo()
        elif choice == "12": print("Goodbye!"); break
        else: print("Choose an option from 1 to 12.")


if __name__ == "__main__":
    try: menu()
    except (KeyboardInterrupt, EOFError): print("\nGoodbye!")
