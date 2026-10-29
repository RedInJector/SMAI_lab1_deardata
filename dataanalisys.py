#!/usr/bin/env python
# -*- coding: utf-8 -*-

from imgui.integrations.glfw import GlfwRenderer
from testwindow import show_test_window
import OpenGL.GL as gl
import glfw
import imgui
import sys

import pandas as pd
from dataclasses import dataclass, field

import numpy as np

from pathlib import Path


@dataclass
class CleaningReport:
    missing_before: int = 0
    missing_after: int = 0
    duplicates_removed: int = 0
    imputed_columns: list[str] = field(default_factory=list)
    converted_columns: list[str] = field(default_factory=list)

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.original = df.copy()
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
            if not any(
                word in str(column).lower()
                for word in ("date", "timestamp")
            ):
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

CSV_FILE = "data.csv"

df = pd.read_csv(CSV_FILE)
cleaner = DataCleaner(df)
cleaned_df = cleaner.clean()

numeric_columns = cleaned_df.select_dtypes(include="number").columns.tolist()
category_columns = cleaned_df.select_dtypes(
    include=["object", "category", "string", "bool"]
).columns.tolist()

# Initial selections, chosen safely from the available columns.
filter_column_a = numeric_columns[0] if numeric_columns else ""
filter_column_b = (
    numeric_columns[1] if len(numeric_columns) > 1
    else filter_column_a
)
group_column = category_columns[0] if category_columns else ""
aggregate_column = numeric_columns[0] if numeric_columns else ""
aggregate_method = "mean"
ranking_column = numeric_columns[0] if numeric_columns else ""
trend_column = numeric_columns[0] if numeric_columns else ""
filter_value_a = 0.0
filter_value_b = 0.0

numpy_column_a = numeric_columns[0] if numeric_columns else ""
numpy_column_b = (
    numeric_columns[1] if len(numeric_columns) > 1 else numpy_column_a
)
numpy_threshold = 0.0
numpy_category_column = (
    numeric_columns[0] if numeric_columns else ""
)

def draw_numpy_statistics(data: pd.DataFrame) -> None:
    """Task 1: Calculate mean, median, and std using explicit NumPy functions."""
    global numpy_column_a
    imgui.text("1. NumPy Basic Statistics")

    numpy_column_a = draw_combo(
        "Select target column", numeric_columns, numpy_column_a
    )

    if not numpy_column_a or numpy_column_a not in data.columns:
        imgui.text("No valid numerical column selected.")
        return

    # Extract NumPy ndarray directly
    arr = data[numpy_column_a].dropna().to_numpy()

    if arr.size == 0:
        imgui.text("Column contains no valid numeric values.")
        return

    # Task 1 NumPy operations
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    std_val = np.std(arr)

    imgui.bullet_text(f"np.mean(): {mean_val:.4f}")
    imgui.bullet_text(f"np.median(): {median_val:.4f}")
    imgui.bullet_text(f"np.std(): {std_val:.4f}")


def draw_numpy_normalization(data: pd.DataFrame) -> None:
    """Task 2: Normalize numerical values using Min-Max normalization."""
    imgui.text("2. Min-Max Normalization (NumPy)")

    if not numpy_column_a or numpy_column_a not in data.columns:
        return

    arr = data[numpy_column_a].dropna().to_numpy()
    min_val = np.min(arr)
    max_val = np.max(arr)

    if max_val == min_val:
        imgui.text("Cannot normalize: min and max values are equal.")
        return

    # Task 2 Min-Max formula: (x - min) / (max - min)
    normalized_arr = (arr - min_val) / (max_val - min_val)

    imgui.bullet_text(f"Original Range: [{min_val:.2f}, {max_val:.2f}]")
    imgui.bullet_text(
        f"Normalized Range: [{np.min(normalized_arr):.2f}, {np.max(normalized_arr):.2f}]"
    )

    # Show snippet preview of first 5 normalized values
    preview_rows = [
        (f"Row {idx}", f"{orig:.4f}", f"{norm:.4f}")
        for idx, (orig, norm) in enumerate(zip(arr[:5], normalized_arr[:5]))
    ]
    draw_table(["Row", "Original Value", "Normalized Value"], preview_rows)


def draw_numpy_conditional(data: pd.DataFrame) -> None:
    """Tasks 3 & 4: Identify threshold values and attach a new column using np.where()."""
    global numpy_threshold, numpy_category_column
    imgui.text("3 & 4. Thresholding & Conditional Column Creation (np.where)")

    numpy_category_column = draw_combo(
        "Target column", numeric_columns, numpy_category_column
    )

    if not numpy_category_column or numpy_category_column not in data.columns:
        return

    changed, numpy_threshold = imgui.input_float(
        "Threshold Value", numpy_threshold
    )

    arr = data[numpy_category_column].to_numpy()

    # Task 3: Identify values above/below threshold
    above_mask = arr > numpy_threshold
    above_count = np.count_nonzero(above_mask)
    below_count = arr.size - above_count

    imgui.bullet_text(f"Values above {numpy_threshold:.2f}: {above_count}")
    imgui.bullet_text(f"Values at or below {numpy_threshold:.2f}: {below_count}")

    # Task 4: Create new categorical indicator column using np.where()
    new_col_name = f"{numpy_category_column}_category"
    data[new_col_name] = np.where(arr > numpy_threshold, "High", "Low")

    # Preview updated DataFrame with newly appended NumPy column
    imgui.text(f"Updated DataFrame preview with column '{new_col_name}':")
    preview_cols = [numpy_category_column, new_col_name]
    preview_data = data[preview_cols].head(5).itertuples(index=False, name=None)
    draw_table(preview_cols, preview_data)


def draw_numpy_correlation(data: pd.DataFrame) -> None:
    """Task 5: Calculate Pearson correlation matrix using np.corrcoef()."""
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

    # Clean missing values jointly across both columns for matrix math
    clean_pair = data[[numpy_column_a, numpy_column_b]].dropna()
    arr_a = clean_pair[numpy_column_a].to_numpy()
    arr_b = clean_pair[numpy_column_b].to_numpy()

    if arr_a.size == 0:
        imgui.text("No valid common records between selected columns.")
        return

    # Task 5: Compute 2x2 correlation matrix
    corr_matrix = np.corrcoef(arr_a, arr_b)
    corr_val = corr_matrix[0, 1]

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
    """Main section handler for Part D."""
    draw_numpy_statistics(cleaned_df)
    imgui.separator()
    draw_numpy_normalization(cleaned_df)
    imgui.separator()
    draw_numpy_conditional(cleaned_df)
    imgui.separator()
    draw_numpy_correlation(cleaned_df)

def draw_section(title: str, content) -> None:
    if imgui.collapsing_header(title)[0]:
        content()


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

    imgui.columns(len(preview.columns), "preview", True)

    for column in preview.columns:
        imgui.text_unformatted(str(column))
        imgui.next_column()

    imgui.separator()

    for row in preview.itertuples(index=False, name=None):
        for value in row:
            imgui.text_unformatted(str(value))
            imgui.next_column()

    imgui.columns(1)


def format_cell(value, decimals: int = 4) -> str:
    """Safely format floats to fixed decimals, while keeping integers and text untouched."""
    try:
        val_float = float(value)

        # If it's a whole number (e.g. 4.0, 100.0), display as an integer without decimals
        if val_float.is_integer():
            return f"{int(val_float)}"

        # Otherwise format with trailing zeros stripped (or fixed decimals)
        return f"{val_float:.{decimals}f}".rstrip('0').rstrip('.')

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


def draw_overview(data: pd.DataFrame) -> None:
    imgui.text(f"Rows: {data.shape[0]}")
    imgui.text(f"Columns: {data.shape[1]}")


def draw_column_info(data: pd.DataFrame) -> None:
    metadata = pd.DataFrame({
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str).values,
        "Missing Values": data.isna().sum().values,
    })

    draw_table(
        ["Column", "Data Type", "Missing Values"],
        metadata.itertuples(index=False, name=None),
    )

def draw_combo(label: str, options: list[str], current: str) -> str:
    """Display a combo box and return the selected option."""
    if not options:
        imgui.text(f"No options available for {label}.")
        return current

    try:
        index = options.index(current)
    except ValueError:
        index = 0

    changed, index = imgui.combo(label, index, options)
    return options[index]

def explore_data() -> None:

    imgui.text("Dataset Overview")

    draw_overview(df)

    imgui.text("First Five Rows")
    draw_table(
        df.columns.tolist(),
        df.head().itertuples(index=False, name=None),
    )

    imgui.text("Column Information")
    draw_column_info(df)

def draw_summary() -> None:
    imgui.text("Summary Statistics")

    if not numeric_columns:
        imgui.text("No numerical columns found.")
        return

    summary = df[numeric_columns].describe().T.reset_index()
    summary = summary.rename(columns={"index": "Column"})

    draw_table(
        summary.columns.tolist(),
        summary.itertuples(index=False, name=None),
    )


def draw_filter() -> None:
    global filter_column_a, filter_column_b
    global filter_value_a, filter_value_b

    imgui.text("Filter Rows (two conditions)")

    if len(numeric_columns) < 2:
        imgui.text("At least two numerical columns are required.")
        return

    global filter_column_a, filter_column_b

    filter_column_a = draw_combo(
        "First column", numeric_columns, filter_column_a
    )
    filter_column_b = draw_combo(
        "Second column", numeric_columns, filter_column_b
    )

    changed, filter_value_a = imgui.input_float(
        "Minimum " + filter_column_a, filter_value_a
    )
    changed, filter_value_b = imgui.input_float(
        "Minimum " + filter_column_b, filter_value_b
    )

    filtered = df[
        (df[filter_column_a] >= filter_value_a)
        & (df[filter_column_b] >= filter_value_b)
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

    group_column = draw_combo(
        "Group by", category_columns, group_column
    )
    aggregate_column = draw_combo(
        "Aggregate column", numeric_columns, aggregate_column
    )
    aggregate_method = draw_combo(
        "Aggregation",
        ["mean", "sum", "count"],
        aggregate_method,
    )

    grouped = (
        df.groupby(group_column, dropna=False)[aggregate_column]
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

    ranking_column = draw_combo(
        "Rank by", numeric_columns, ranking_column
    )

    top = df.nlargest(5, ranking_column)

    draw_table(
        top.columns.tolist(),
        top.itertuples(index=False, name=None),
    )


def draw_trend() -> None:
    imgui.text("Trend / Pattern")

    if not numeric_columns:
        imgui.text("No numerical columns available for trend analysis.")
        return

    global trend_column

    trend_column = draw_combo(
        "Analyze distribution", numeric_columns, trend_column
    )

    values = df[trend_column].dropna()

    if values.empty:
        imgui.text("No valid values to analyze.")
        return

    mean = values.mean()
    median = values.median()
    std = values.std()

    imgui.text(f"Mean: {mean:.3f}")
    imgui.text(f"Median: {median:.3f}")
    imgui.text(f"Standard deviation: {std:.3f}")

    if mean > median:
        imgui.text("Pattern: mean > median; possible right-skew.")
    elif mean < median:
        imgui.text("Pattern: mean < median; possible left-skew.")
    else:
        imgui.text("Pattern: mean and median are approximately equal.")

    if std == 0:
        imgui.text("All valid values are identical.")
    elif std < abs(mean) * 0.1:
        imgui.text("Values show low variation relative to the mean.")
    else:
        imgui.text("Values show measurable variation.")




def draw_data_analisys():
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


def DrawUI() -> None:
    viewport = imgui.get_main_viewport()

    # Position window at top-left corner of viewport
    imgui.set_next_window_position(viewport.pos.x, viewport.pos.y)
    # Set window size to match full viewport width & height
    imgui.set_next_window_size(viewport.size.x, viewport.size.y)

    # Window flags to remove title bar, borders, and resizing grips
    flags = (
            imgui.WINDOW_NO_RESIZE |
            imgui.WINDOW_NO_MOVE |
            imgui.WINDOW_NO_COLLAPSE |
            imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS |
            imgui.WINDOW_NO_COLLAPSE
    )

    imgui.begin(CSV_FILE, flags=flags)

    draw_section("Explore Data", explore_data)
    draw_section("Cleaning Report", draw_report)
    draw_section("Cleaned Data Preview", draw_preview)
    draw_section("Data Analysis", draw_data_analisys)
    draw_section("Numpy Analysis", draw_numpy_analysis)

    imgui.end()

def main():
    imgui.create_context()
    window = impl_glfw_init()
    impl = GlfwRenderer(window)

    show_custom_window = True

    glfw.swap_interval(1)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        impl.process_inputs()

        imgui.new_frame()

        show_test_window()

        DrawUI()

        gl.glClearColor(0.1, 0.1, 0.1, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        imgui.render()
        impl.render(imgui.get_draw_data())
        glfw.swap_buffers(window)

    impl.shutdown()
    glfw.terminate()


def impl_glfw_init():
    width, height = 1280, 720
    window_name = "Data Explorer"

    if not glfw.init():
        print("Could not initialize OpenGL context")
        sys.exit(1)

    # OS X supports only forward-compatible core profiles from 3.2
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    glfw.make_context_current(window)

    if not window:
        glfw.terminate()
        print("Could not initialize Window")
        sys.exit(1)

    return window


if __name__ == "__main__":
    main()
