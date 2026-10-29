#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import sys

from imgui.integrations.glfw import GlfwRenderer
import glfw
import imgui
import numpy as np
import OpenGL.GL as gl
import pandas as pd
from testwindow import show_test_window

CSV_FILE = "data.csv"


@dataclass
class CleaningReport:
    missing_before: int = 0
    missing_after: int = 0
    duplicates_removed: int = 0
    imputed_columns: list[str] = field(default_factory=list)
    converted_columns: list[str] = field(default_factory=list)

@dataclass
class ExecutiveSummary:
    # E.1 Metrics
    total_rows: int = 0
    total_cols: int = 0
    total_imputed: int = 0
    top_column: str = ""
    top_column_mean: float = 0.0

    # E.2 Report
    report_text: str = ""

    # E.3 Plain English Findings (generated dynamically)
    findings: list[tuple[str, str]] = field(default_factory=list)

    # E.4 Dynamic Conclusion
    conclusion_text: str = ""

class DataCleaner:

    def __init__(self, df: pd.DataFrame):
        self.data = df.copy()
        self.report = CleaningReport()

    def handle_missing_values(self) -> None:
        self.report.missing_before = int(self.data.isna().sum().sum())

        for column in self.data.columns:
            series = self.data[column]
            if not series.isna().any():
                continue

            if pd.api.types.is_numeric_dtype(series):
                value = series.median()
            else:
                modes = series.mode()
                value = modes.iloc[0] if not modes.empty else None

            if value is not None and not pd.isna(value):
                self.data[column] = series.fillna(value)
                self.report.imputed_columns.append(str(column))

        self.report.missing_after = int(self.data.isna().sum().sum())

    def convert_types(self) -> None:
        for column in self.data.columns:
            col_str = str(column).lower()
            if not any(word in col_str for word in ("date", "timestamp")):
                continue

            converted = pd.to_datetime(self.data[column], errors="coerce")
            if converted.notna().any():
                self.data[column] = converted
                self.report.converted_columns.append(str(column))

    def remove_duplicates(self) -> None:
        self.report.duplicates_removed = int(self.data.duplicated().sum())
        self.data = self.data.drop_duplicates().reset_index(drop=True)

    def clean(self) -> pd.DataFrame:
        self.handle_missing_values()
        self.remove_duplicates()
        self.convert_types()
        return self.data

def generate_executive_summary(
    df: pd.DataFrame, cleaner: DataCleaner
) -> ExecutiveSummary:
    """Computes all stats and dynamic text strings ONCE."""
    summary = ExecutiveSummary()
    summary.total_rows = len(df)
    summary.total_cols = len(df.columns)
    summary.total_imputed = len(cleaner.report.imputed_columns)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    report = cleaner.report

    # --- E.2 Dynamic Text Report ---
    summary.report_text = (
        f"Data Processing Summary for Dataset ({summary.total_rows} rows x {summary.total_cols} columns):\n"
        f"- Missing Values: {report.missing_before} original missing values were detected. "
        f"After processing, {report.missing_after} missing values remain.\n"
        f"- Duplicates: {report.duplicates_removed} duplicate rows were identified and purged.\n"
        f"- Imputation: Applied median imputation on numeric columns ({', '.join(report.imputed_columns) if report.imputed_columns else 'None'}) "
        f"and mode imputation on categorical fields."
    )

    # --- E.3 Dynamic Findings Generator ---
    findings = []

    # Finding 1: Top numeric column driver
    if numeric_cols:
        # Find column with highest average value
        means = df[numeric_cols].mean()
        top_col = means.idxmax()
        top_val = means.max()
        summary.top_column = top_col
        summary.top_column_mean = top_val

        findings.append((
            f"1. Leading Numerical Feature ({top_col}):",
            f"Across all {len(numeric_cols)} numeric attributes, '{top_col}' holds the highest average value at {top_val:.2f}.",
        ))

    # Finding 2: Skewness detection using mean vs median comparison
    if numeric_cols:
        skewed_cols = []
        for col in numeric_cols:
            col_mean = df[col].mean()
            col_median = df[col].median()
            if abs(col_mean - col_median) > 0.1 * abs(col_mean or 1):
                direction = "right-skewed" if col_mean > col_median else "left-skewed"
                skewed_cols.append((col, direction))

        if skewed_cols:
            target_col, skew_dir = skewed_cols[0]
            findings.append((
                "2. Distribution Asymmetry:",
                f"Column '{target_col}' shows significant asymmetric distribution ({skew_dir}), "
                f"where the mean ({df[target_col].mean():.2f}) deviates noticeably from the median ({df[target_col].median():.2f}).",
            ))
        else:
            findings.append((
                "2. Symmetric Distributions:",
                "Numerical attributes exhibit symmetrical distributions where mean and median values closely align.",
            ))

    # Finding 3: Dynamic Category Analysis
    if cat_cols and numeric_cols:
        group_col = cat_cols[0]
        num_col = numeric_cols[0]
        grouped = df.groupby(group_col)[num_col].mean()
        max_cat = grouped.idxmax()
        max_val = grouped.max()

        findings.append((
            f"3. High-Performing Segment in '{group_col}':",
            f"Grouping by '{group_col}' reveals that category '{max_cat}' achieves the highest aggregate average for '{num_col}' ({max_val:.2f}).",
        ))
    else:
        findings.append((
            "3. Data Structure:",
            f"Dataset consists of {len(numeric_cols)} numerical features and {len(cat_cols)} categorical fields available for modeling.",
        ))

    summary.findings = findings

    # --- E.4 Dynamic Conclusion ---
    if numeric_cols:
        summary.conclusion_text = (
            f"What the data reveals: The dataset contains {summary.total_rows} verified records. "
            f"Key metrics are heavily influenced by '{summary.top_column}' (mean = {summary.top_column_mean:.2f}). "
            f"Data quality has been preserved through removal of {report.duplicates_removed} duplicates "
            f"and imputation across {summary.total_imputed} features."
        )
    else:
        summary.conclusion_text = "What the data reveals: Dataset processing completed successfully with zero numeric columns detected."

    return summary

# --- Data Initialization ---
raw_df = pd.read_csv(CSV_FILE)
cleaner = DataCleaner(raw_df)
cleaned_df = cleaner.clean()

numeric_columns = cleaned_df.select_dtypes(include="number").columns.tolist()
category_columns = cleaned_df.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()

# Global widget states
filter_column_a = numeric_columns[0] if numeric_columns else ""
filter_column_b = (
    numeric_columns[1] if len(numeric_columns) > 1 else filter_column_a
)
filter_value_a = 0.0
filter_value_b = 0.0

group_column = category_columns[0] if category_columns else ""
aggregate_column = numeric_columns[0] if numeric_columns else ""
aggregate_method = "mean"

ranking_column = numeric_columns[0] if numeric_columns else ""
trend_column = numeric_columns[0] if numeric_columns else ""

numpy_column_a = numeric_columns[0] if numeric_columns else ""
numpy_column_b = (
    numeric_columns[1] if len(numeric_columns) > 1 else numpy_column_a
)
numpy_threshold = 0.0
numpy_category_column = numeric_columns[0] if numeric_columns else ""

exec_summary = generate_executive_summary(cleaned_df, cleaner)


def format_cell(value, decimals: int = 4) -> str:
    """Format floats with precision while keeping whole integers and strings clean."""
    try:
        val_float = float(value)
        if val_float.is_integer():
            return f"{int(val_float)}"
        return f"{val_float:.{decimals}f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        return str(value)

def draw_table(headers: list[str], rows) -> None:
    """Render tabular data using ImGui's columns API."""
    if not headers:
        imgui.text("No columns to display.")
        return

    imgui.columns(len(headers), "##data_table", True)

    for header in headers:
        imgui.text_unformatted(str(header))
        imgui.next_column()

    imgui.separator()

    for row in rows:
        for value in row:
            imgui.text_unformatted(format_cell(value, decimals=4))
            imgui.next_column()

    imgui.columns(1)

def draw_combo(label: str, options: list[str], current: str) -> str:
    """Display a combo box and return the selected option."""
    if not options:
        imgui.text(f"No options available for {label}.")
        return current

    try:
        index = options.index(current)
    except ValueError:
        index = 0

    _, index = imgui.combo(label, index, options)
    return options[index]

def draw_section(title: str, content) -> None:
    if imgui.collapsing_header(title)[0]:
        content()

def draw_overview(data: pd.DataFrame) -> None:
    imgui.text(f"Rows: {data.shape[0]}")
    imgui.text(f"Columns: {data.shape[1]}")

def draw_column_info(data: pd.DataFrame) -> None:
    metadata = pd.DataFrame(
        {
            "Column": data.columns,
            "Data Type": data.dtypes.astype(str).values,
            "Missing Values": data.isna().sum().values,
        }
    )
    draw_table(
        ["Column", "Data Type", "Missing Values"],
        metadata.itertuples(index=False, name=None),
    )

def explore_data() -> None:
    imgui.text("Dataset Overview")
    draw_overview(cleaned_df)

    imgui.text("First Five Rows")
    draw_table(
        cleaned_df.columns.tolist(),
        cleaned_df.head().itertuples(index=False, name=None),
    )

    imgui.text("Column Information")
    draw_column_info(cleaned_df)

def draw_report() -> None:
    report = cleaner.report

    imgui.text("Missing values")
    imgui.bullet_text(f"Before: {report.missing_before}")
    imgui.bullet_text(f"After: {report.missing_after}")

    imgui.separator()
    imgui.text("Duplicates removed")
    imgui.bullet_text(str(report.duplicates_removed))

    imgui.separator()
    imgui.text("Imputed columns")
    for column in report.imputed_columns:
        imgui.bullet_text(column)

    imgui.separator()
    imgui.text("Converted date columns")
    for column in report.converted_columns:
        imgui.bullet_text(column)

def draw_preview() -> None:
    preview = cleaned_df.head()
    if preview.empty:
        imgui.text("No rows to display.")
        return
    draw_table(
        preview.columns.tolist(), preview.itertuples(index=False, name=None)
    )

def draw_summary() -> None:
    imgui.text("Summary Statistics")
    if not numeric_columns:
        imgui.text("No numerical columns found.")
        return

    summary = cleaned_df[numeric_columns].describe().T.reset_index()
    summary = summary.rename(columns={"index": "Column"})
    draw_table(
        summary.columns.tolist(), summary.itertuples(index=False, name=None)
    )

def draw_filter() -> None:
    global filter_column_a, filter_column_b, filter_value_a, filter_value_b

    imgui.text("Filter Rows (two conditions)")

    if len(numeric_columns) < 2:
        imgui.text("At least two numerical columns are required.")
        return

    filter_column_a = draw_combo(
        "First column", numeric_columns, filter_column_a
    )
    filter_column_b = draw_combo(
        "Second column", numeric_columns, filter_column_b
    )

    _, filter_value_a = imgui.input_float(
        "Minimum " + filter_column_a, filter_value_a
    )
    _, filter_value_b = imgui.input_float(
        "Minimum " + filter_column_b, filter_value_b
    )

    filtered = cleaned_df[
        (cleaned_df[filter_column_a] >= filter_value_a)
        & (cleaned_df[filter_column_b] >= filter_value_b)
    ]

    imgui.text(f"Matching rows: {len(filtered)}")
    draw_table(
        filtered.columns.tolist(),
        filtered.head(10).itertuples(index=False, name=None),
    )

def draw_group_analysis() -> None:
    imgui.text("Group and Aggregate")

    if not category_columns or not numeric_columns:
        imgui.text("Requires a categorical and a numerical column.")
        return

    global group_column, aggregate_column, aggregate_method

    group_column = draw_combo("Group by", category_columns, group_column)
    aggregate_column = draw_combo(
        "Aggregate column", numeric_columns, aggregate_column
    )
    aggregate_method = draw_combo(
        "Aggregation", ["mean", "sum", "count"], aggregate_method
    )

    grouped = (
        cleaned_df.groupby(group_column, dropna=False)[aggregate_column]
        .agg(aggregate_method)
        .reset_index()
        .sort_values(aggregate_column, ascending=False)
    )

    draw_table(
        grouped.columns.tolist(),
        grouped.head(20).itertuples(index=False, name=None),
    )

def draw_top_five() -> None:
    imgui.text("Top Five Records")

    if not numeric_columns:
        imgui.text("No numerical columns found.")
        return

    global ranking_column
    ranking_column = draw_combo("Rank by", numeric_columns, ranking_column)

    top = cleaned_df.nlargest(5, ranking_column)
    draw_table(top.columns.tolist(), top.itertuples(index=False, name=None))

def draw_trend() -> None:
    imgui.text("Trend / Pattern")

    if not numeric_columns:
        imgui.text("No numerical columns available for trend analysis.")
        return

    global trend_column
    trend_column = draw_combo(
        "Analyze distribution", numeric_columns, trend_column
    )

    values = cleaned_df[trend_column].dropna()
    if values.empty:
        imgui.text("No valid values to analyze.")
        return

    mean_val = values.mean()
    median_val = values.median()
    std_val = values.std()

    imgui.text(f"Mean: {mean_val:.3f}")
    imgui.text(f"Median: {median_val:.3f}")
    imgui.text(f"Standard deviation: {std_val:.3f}")

    if mean_val > median_val:
        imgui.text("Pattern: mean > median; possible right-skew.")
    elif mean_val < median_val:
        imgui.text("Pattern: mean < median; possible left-skew.")
    else:
        imgui.text("Pattern: mean and median are approximately equal.")

    if std_val == 0:
        imgui.text("All valid values are identical.")
    elif std_val < abs(mean_val) * 0.1:
        imgui.text("Values show low variation relative to the mean.")
    else:
        imgui.text("Values show measurable variation.")

def draw_data_analysis() -> None:
    imgui.text("1. Summary Statistics")
    draw_summary()

    imgui.text("2. Filter with Two Conditions")
    draw_filter()

    imgui.text("3. Group and Aggregate")
    draw_group_analysis()

    imgui.text("4. Top Five Records")
    draw_top_five()

    imgui.text("5. Trend / Pattern")
    draw_trend()

# --- NumPy Handlers ---
def draw_numpy_statistics(data: pd.DataFrame) -> None:
    global numpy_column_a
    imgui.text("1. NumPy Basic Statistics")

    numpy_column_a = draw_combo(
        "Select target column", numeric_columns, numpy_column_a
    )
    if not numpy_column_a or numpy_column_a not in data.columns:
        imgui.text("No valid numerical column selected.")
        return

    arr = data[numpy_column_a].dropna().to_numpy()
    if arr.size == 0:
        imgui.text("Column contains no valid numeric values.")
        return

    imgui.bullet_text(f"np.mean(): {np.mean(arr):.4f}")
    imgui.bullet_text(f"np.median(): {np.median(arr):.4f}")
    imgui.bullet_text(f"np.std(): {np.std(arr):.4f}")

def draw_numpy_normalization(data: pd.DataFrame) -> None:
    imgui.text("2. Min-Max Normalization (NumPy)")

    if not numpy_column_a or numpy_column_a not in data.columns:
        return

    arr = data[numpy_column_a].dropna().to_numpy()
    min_val, max_val = np.min(arr), np.max(arr)

    if max_val == min_val:
        imgui.text("Cannot normalize: min and max values are equal.")
        return

    normalized_arr = (arr - min_val) / (max_val - min_val)

    imgui.bullet_text(f"Original Range: [{min_val:.2f}, {max_val:.2f}]")
    imgui.bullet_text(
        f"Normalized Range: [{np.min(normalized_arr):.2f}, {np.max(normalized_arr):.2f}]"
    )

    preview_rows = [
        (f"Row {idx}", orig, norm)
        for idx, (orig, norm) in enumerate(zip(arr[:5], normalized_arr[:5]))
    ]
    draw_table(["Row", "Original Value", "Normalized Value"], preview_rows)

def draw_numpy_conditional(data: pd.DataFrame) -> None:
    global numpy_threshold, numpy_category_column
    imgui.text("3 & 4. Thresholding & Conditional Column Creation (np.where)")

    numpy_category_column = draw_combo(
        "Target column", numeric_columns, numpy_category_column
    )
    if not numpy_category_column or numpy_category_column not in data.columns:
        return

    _, numpy_threshold = imgui.input_float("Threshold Value", numpy_threshold)

    arr = data[numpy_category_column].to_numpy()
    above_count = np.count_nonzero(arr > numpy_threshold)
    below_count = arr.size - above_count

    imgui.bullet_text(f"Values above {numpy_threshold:.2f}: {above_count}")
    imgui.bullet_text(
        f"Values at or below {numpy_threshold:.2f}: {below_count}"
    )

    new_col_name = f"{numpy_category_column}_category"
    data[new_col_name] = np.where(arr > numpy_threshold, "High", "Low")

    imgui.text(f"Updated DataFrame preview with column '{new_col_name}':")
    preview_cols = [numpy_category_column, new_col_name]
    draw_table(
        preview_cols,
        data[preview_cols].head(5).itertuples(index=False, name=None),
    )

def draw_numpy_correlation(data: pd.DataFrame) -> None:
    global numpy_column_a, numpy_column_b
    imgui.text("5. Correlation Analysis (np.corrcoef)")

    if len(numeric_columns) < 2:
        imgui.text("Requires at least two numeric columns.")
        return

    numpy_column_a = draw_combo("Column A", numeric_columns, numpy_column_a)
    numpy_column_b = draw_combo("Column B", numeric_columns, numpy_column_b)

    if numpy_column_a == numpy_column_b:
        imgui.text("Select two distinct columns to calculate correlation.")
        return

    clean_pair = data[[numpy_column_a, numpy_column_b]].dropna()
    arr_a = clean_pair[numpy_column_a].to_numpy()
    arr_b = clean_pair[numpy_column_b].to_numpy()

    if arr_a.size == 0:
        imgui.text("No valid common records between selected columns.")
        return

    corr_val = np.corrcoef(arr_a, arr_b)[0, 1]
    imgui.bullet_text(f"Pearson Correlation Coefficient: {corr_val:.4f}")

    if np.isnan(corr_val):
        imgui.text("Correlation is undefined (zero variance).")
    elif abs(corr_val) > 0.7:
        imgui.text("Pattern: Strong linear relationship detected.")
    elif abs(corr_val) > 0.3:
        imgui.text("Pattern: Moderate linear relationship detected.")
    else:
        imgui.text("Pattern: Weak or no linear relationship detected.")

def draw_numpy_analysis() -> None:
    draw_numpy_statistics(cleaned_df)
    imgui.separator()
    draw_numpy_normalization(cleaned_df)
    imgui.separator()
    draw_numpy_conditional(cleaned_df)
    imgui.separator()
    draw_numpy_correlation(cleaned_df)

def draw_results_presentation() -> None:
    """Part E: Present dynamic summary metrics, text reports, findings, and conclusions."""

    # E.1: At least three useful summary metrics (Dynamic)
    imgui.text_colored("Executive Summary Metrics", 0.4, 0.8, 1.0)

    imgui.columns(3, "##summary_metrics", False)

    imgui.text("Total Dataset Size")
    imgui.bullet_text(f"{exec_summary.total_rows} Rows")
    imgui.next_column()

    imgui.text("Primary Driver")
    imgui.bullet_text(f"{exec_summary.top_column}")
    imgui.next_column()

    imgui.text("Data Quality")
    imgui.bullet_text(f"{exec_summary.total_imputed} Cols Imputed")
    imgui.next_column()

    imgui.columns(1)
    imgui.separator()

    # E.2: Dynamic Text-Based Report
    if imgui.collapsing_header("Detailed Data Cleaning & Profile Report")[0]:
        imgui.text_wrapped(exec_summary.report_text)

    # E.3: Dynamic Plain English Findings
    if imgui.collapsing_header("Key Dataset Findings")[0]:
        imgui.text_colored("Key Analytical Insights:", 0.2, 0.9, 0.4)

        for title, detail in exec_summary.findings:
            imgui.bullet_text(title)
            imgui.indent()
            imgui.text_wrapped(detail)
            imgui.unindent()

    # E.4: Dynamic Conclusion
    if imgui.collapsing_header("Conclusion")[0]:
        imgui.text_colored("Summary & Practical Takeaways", 0.9, 0.8, 0.3)
        imgui.text_wrapped(exec_summary.conclusion_text)


def draw_ui() -> None:
    viewport = imgui.get_main_viewport()

    imgui.set_next_window_position(viewport.pos.x, viewport.pos.y)
    imgui.set_next_window_size(viewport.size.x, viewport.size.y)

    flags = (
        imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_NO_COLLAPSE
        | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
    )

    imgui.begin(CSV_FILE, flags=flags)

    draw_section("Part A - Explore Data", explore_data)
    draw_section("Part B - Cleaning Report", draw_report)
    draw_section("Cleaned Data Preview", draw_preview)
    draw_section("Part C - Data Analysis", draw_data_analysis)
    draw_section("Part D - Numpy Analysis", draw_numpy_analysis)
    draw_section("Part E - Present Results", draw_results_presentation)

    imgui.end()

def impl_glfw_init():
    width, height = 1280, 720
    window_name = "Data Explorer"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    window = glfw.create_window(
        int(width), int(height), window_name, None, None
    )
    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    glfw.make_context_current(window)
    return window

def main():
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)

    glfw.swap_interval(1)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        show_test_window()
        draw_ui()

        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()

if __name__ == "__main__":
    main()