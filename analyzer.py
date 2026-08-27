import pandas as pd
import numpy as np
from datetime import datetime


def make_json_safe(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value]

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    return value


def load_dataset(file_path: str) -> pd.DataFrame:
    file_path_lower = file_path.lower()

    if file_path_lower.endswith(".csv"):
        encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
        last_error = None
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                if isinstance(df, pd.DataFrame):
                    return df
            except UnicodeDecodeError as error:
                last_error = error
        raise ValueError("Could not read the CSV file. Encoding not supported.") from last_error

    if file_path_lower.endswith(".xlsx"):
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Excel file did not produce a DataFrame.")
            return df
        except Exception as error:
            raise ValueError(f"Could not read XLSX file: {error}") from error

    if file_path_lower.endswith(".xls"):
        try:
            df = pd.read_excel(file_path)
            if not isinstance(df, pd.DataFrame):
                raise TypeError("Excel file did not produce a DataFrame.")
            return df
        except Exception as error:
            raise ValueError(f"Could not read XLS file: {error}") from error

    raise ValueError("Only CSV, XLSX and XLS files are supported.")


def safe_number(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return round(float(value), 4)
    return value


def _standardize_categories(series: pd.Series) -> pd.Series:
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return series

    non_nulls = series.dropna().astype(str).str.strip()
    if non_nulls.empty:
        return series

    unique_vals = set(non_nulls.str.lower())

    gender_map = {
        "m": "Male", "male": "Male", "man": "Male",
        "f": "Female", "female": "Female", "woman": "Female",
        "other": "Other", "o": "Other"}
    if unique_vals.issubset(set(gender_map.keys())):
        return series.astype(str).str.lower().map(gender_map).fillna(series)

    bool_map = {
        "yes": "Yes", "y": "Yes", "true": "Yes", "t": "Yes", "1": "Yes",
        "no": "No", "n": "No", "false": "No", "f": "No", "0": "No"}
    if unique_vals.issubset(set(bool_map.keys())):
        return series.astype(str).str.lower().map(bool_map).fillna(series)

    return series


def _normalize_text_series(series: pd.Series):
    missing_tokens = {
        "", "na", "n/a", "n.a", "n.a.", "nan", "none",
        "null", "nil", "missing", "not available", "not_applicable",
        "not applicable", "-", "--", "?"}
    result = series.astype("string")
    result = result.str.replace(r"[\x00-\x1f\x7f]", "", regex=True)
    result = result.str.replace(r"\s+", " ", regex=True).str.strip()
    lowered = result.str.lower()
    result = result.mask(lowered.isin(missing_tokens), pd.NA)
    return result


def _looks_like_identifier(column_name: str, series: pd.Series) -> bool:
    name = str(column_name).lower().strip()
    id_words = (
        "id", "code", "zip", "postal", "pincode", "pin",
        "phone", "mobile", "account", "invoice", "order",
        "employee", "customer", "product", "sku")
    if any(word == name or name.startswith(word + "_") or name.endswith("_" + word) for word in id_words):
        return True
    non_missing = series.dropna().astype(str).str.strip()
    if non_missing.empty:
        return False
    sample = non_missing.head(500)
    leading_zero_ratio = sample.str.match(r"^0\d+$").mean()
    return bool(leading_zero_ratio >= 0.50)


def _try_numeric(series: pd.Series, column_name: str):
    if _looks_like_identifier(column_name, series):
        return None, 0.0
    text = series.astype("string").str.strip()
    non_missing = text.dropna()
    if non_missing.empty:
        return None, 0.0

    normalized = (
        text
        .str.replace(",", "", regex=False)
        .str.replace(r"^[₹$€£]\s*", "", regex=True)
        .str.replace(r"\s+", "", regex=True))
    numeric = pd.to_numeric(normalized, errors="coerce")
    ratio = float(numeric.notna().sum() / len(non_missing))
    return numeric, ratio


def _try_date(series: pd.Series, column_name: str):
    non_missing = series.dropna()
    if non_missing.empty:
        return None, 0.0

    name = str(column_name).lower()
    date_words = (
        "date", "time", "dob", "birth", "created", "updated",
        "timestamp", "datetime", "deadline", "joined", "start", "end")
    name_hint = any(word in name for word in date_words)
    sample = non_missing.astype(str).head(500)
    date_pattern = sample.str.contains(
        r"[-/:]|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
        case=False, regex=True, na=False)

    if not name_hint and date_pattern.mean() < 0.50:
        return None, 0.0

    try:
        converted = pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        try:
            converted = pd.to_datetime(series, errors="coerce")
        except Exception:
            return None, 0.0

    ratio = float(converted.notna().sum() / len(non_missing))
    threshold = 0.70 if name_hint else 0.90
    if ratio >= threshold:
        return converted, ratio
    return None, ratio


def clean_dataset(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("clean_dataset() requires a pandas DataFrame.")

    cleaned_df = df.copy(deep=True)
    original_rows = int(len(cleaned_df))
    original_columns = int(len(cleaned_df.columns))
    original_missing = int(cleaned_df.isna().sum().sum())

    cleaned_columns = []
    for column in cleaned_df.columns:
        name = str(column).replace("\ufeff", "").strip()
        name = " ".join(name.split())
        cleaned_columns.append(name or "Unnamed_Column")

    seen = {}
    final_columns = []
    for name in cleaned_columns:
        count = seen.get(name, 0)
        final_columns.append(name if count == 0 else f"{name}_{count}")
        seen[name] = count + 1
    cleaned_df.columns = final_columns

    duplicates_to_drop = []
    seen_cols = {}
    for col in cleaned_df.columns:
        col_series = cleaned_df[col]
        # Check if another column has identical content
        is_dup = False
        for prev_col, prev_series in seen_cols.items():
            if col_series.equals(prev_series):
                duplicates_to_drop.append(col)
                is_dup = True
                break
        if not is_dup:
            seen_cols[col] = col_series
    
    if duplicates_to_drop:
        cleaned_df = cleaned_df.drop(columns=duplicates_to_drop)

    for column in cleaned_df.columns:
        series = cleaned_df[column]
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            cleaned_df[column] = _normalize_text_series(series)
            cleaned_df[column] = _standardize_categories(cleaned_df[column])

    cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)

    rows_before = len(cleaned_df)
    cleaned_df = cleaned_df.dropna(how="all")
    empty_rows_removed = int(rows_before - len(cleaned_df))

    cols_before = len(cleaned_df.columns)
    cleaned_df = cleaned_df.dropna(axis=1, how="all")
    empty_columns_removed = int(cols_before - len(cleaned_df.columns))

    numeric_converted_columns = []
    date_converted_columns = []
    percentage_columns = []

    for column in list(cleaned_df.columns):
        series = cleaned_df[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        numeric_values, numeric_ratio = _try_numeric(series, column)
        non_missing_text = series.dropna().astype(str).str.strip()
        percent_ratio = 0.0
        if not non_missing_text.empty:
            percent_ratio = float(non_missing_text.str.match(r"^-?\s*\d+(?:\.\d+)?\s*%$").mean())

        if percent_ratio >= 0.90 and not _looks_like_identifier(column, series):
            percent_values = pd.to_numeric(
                series.astype("string").str.replace("%", "", regex=False).str.strip(),
                errors="coerce",) / 100.0
            cleaned_df[column] = percent_values
            percentage_columns.append(str(column))
            numeric_converted_columns.append(str(column))
            continue

        if numeric_ratio >= 0.90 and numeric_values is not None:
            cleaned_df[column] = numeric_values
            numeric_converted_columns.append(str(column))
            continue

        date_values, date_ratio = _try_date(series, column)
        if date_values is not None:
            cleaned_df[column] = date_values
            date_converted_columns.append(str(column))

    numeric_columns = list(cleaned_df.select_dtypes(include=np.number).columns)
    for column in numeric_columns:
        cleaned_df[column] = pd.to_numeric(cleaned_df[column], errors="coerce")
        cleaned_df[column] = cleaned_df[column].replace([np.inf, -np.inf], np.nan)

    duplicate_count = int(cleaned_df.duplicated().sum())
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    numeric_filled_columns = []
    categorical_filled_columns = []
    date_filled_columns = []
    missing_before_fill = int(cleaned_df.isna().sum().sum())

    for column in cleaned_df.columns:
        series = cleaned_df[column]
        if not series.isna().any():
            continue

        if pd.api.types.is_numeric_dtype(series):
            median = series.median(skipna=True)
            if pd.notna(median):
                cleaned_df[column] = series.fillna(median)
                numeric_filled_columns.append(str(column))
        elif pd.api.types.is_datetime64_any_dtype(series):
            current_date_str = datetime.now().strftime("%d/%m/%Y")
            cleaned_df[column] = pd.to_datetime(series.fillna(current_date_str), format="%d/%m/%Y", errors="coerce")
            date_filled_columns.append(str(column))
        else:
            mode = series.mode(dropna=True)
            if not mode.empty:
                cleaned_df[column] = series.fillna(mode.iloc[0])
                categorical_filled_columns.append(str(column))
            else:
                cleaned_df[column] = series.fillna("Unknown")
                categorical_filled_columns.append(str(column))

    cleaned_df = cleaned_df.dropna(how="all")
    cleaned_df = cleaned_df.dropna(axis=1, how="all")
    cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)

    final_missing = int(cleaned_df.isna().sum().sum())
    final_duplicates = int(cleaned_df.duplicated().sum())

    cleaning = {
        "original_rows": original_rows,
        "original_columns": original_columns,
        "final_rows": int(len(cleaned_df)),
        "final_columns": int(len(cleaned_df.columns)),
        "rows_removed": int(original_rows - len(cleaned_df)),
        "columns_removed": int(original_columns - len(cleaned_df.columns)),
        "empty_rows_removed": empty_rows_removed,
        "empty_columns_removed": empty_columns_removed,
        "duplicate_columns_dropped": duplicates_to_drop,
        "duplicates_removed": duplicate_count,
        "original_missing_values": original_missing,
        "missing_values_before_fill": missing_before_fill,
        "remaining_missing_values": final_missing,
        "remaining_duplicates": final_duplicates,
        "numeric_columns_converted": numeric_converted_columns,
        "percentage_columns_converted": percentage_columns,
        "date_columns_converted": date_converted_columns,
        "numeric_missing_values_filled": numeric_filled_columns,
        "categorical_missing_values_filled": categorical_filled_columns,
        "date_missing_values_filled": date_filled_columns,
        "quality_flags": {"outliers_removed": 0},}

    return cleaned_df, make_json_safe(cleaning)


def detect_column_types(df):
    numeric, categorical, dates = [], [], []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric.append(str(column))
        elif pd.api.types.is_datetime64_any_dtype(series):
            dates.append(str(column))
        else:
            categorical.append(str(column))
    return make_json_safe({
        "numeric": numeric,
        "categorical": categorical,
        "date": dates})


def column_details(df):
    results = []
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))

        if pd.api.types.is_numeric_dtype(series):
            details = {
                "min": safe_number(series.min()),
                "max": safe_number(series.max()),
                "mean": safe_number(series.mean()),
                "median": safe_number(series.median()),
                "std": safe_number(series.std())}
            column_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            minimum = series.min()
            maximum = series.max()
            details = {
                "min": None if pd.isna(minimum) else minimum.strftime("%d/%m/%Y"),
                "max": None if pd.isna(maximum) else maximum.strftime("%d/%m/%Y")}
            column_type = "date"
        else:
            top_values = series.fillna("Unknown").astype(str).value_counts().head(10)
            details = {
                "top_values": {
                    str(key): int(value)
                    for key, value in top_values.items()}}
            column_type = "categorical"

        results.append(make_json_safe({
            "name": str(column),
            "type": column_type,
            "missing": missing,
            "unique": unique,
            **details}))
    return results


def make_preview(df, rows=15):
    preview = df.head(rows).copy()
    for column in preview.columns:
        if pd.api.types.is_datetime64_any_dtype(preview[column]):
            preview[column] = preview[column].apply(
                lambda x: x.strftime("%d/%m/%Y") if pd.notna(x) else None)
    preview = preview.where(pd.notna(preview), None)
    return make_json_safe(preview.to_dict(orient="records"))


def generate_kpis(df):
    kpis = {
        "Total Records": int(len(df)),
        "Total Columns": int(len(df.columns))}
    numeric_columns = df.select_dtypes(include=np.number).columns
    for column in numeric_columns[:8]:
        series = df[column].dropna()
        if len(series) == 0:
            continue
        kpis[f"Average {column}"] = round(float(series.mean()), 2)
    return make_json_safe(kpis)


def generate_categories(df):
    categorical_columns = df.select_dtypes(include=["object", "string", "category"]).columns
    if len(categorical_columns) == 0:
        return None

    selected_column = None
    for column in categorical_columns:
        unique_count = df[column].nunique(dropna=True)
        if 2 <= unique_count <= 30:
            selected_column = column
            break

    if selected_column is None:
        return None

    counts = df[selected_column].fillna("Unknown").astype(str).value_counts().head(15)
    return make_json_safe({
        "labels": [str(x) for x in counts.index],
        "values": [int(x) for x in counts.values],
        "category_column": str(selected_column),
        "metric_column": None})


def generate_distribution(df):
    numeric_columns = df.select_dtypes(include=np.number).columns
    if len(numeric_columns) == 0:
        return None

    selected_column = numeric_columns[0]
    series = df[selected_column].dropna()
    if len(series) == 0:
        return None

    if series.nunique() <= 20:
        counts = series.value_counts().sort_index()
        return make_json_safe({
            "label": str(selected_column),
            "labels": [str(x) for x in counts.index],
            "values": [int(x) for x in counts.values]})

    counts, bins = np.histogram(series, bins=20)
    labels = [f"{bins[i]:.1f} - {bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    return make_json_safe({
        "label": str(selected_column),
        "labels": labels,
        "values": [int(x) for x in counts]})


def generate_insights(df, cleaning):
    insights = []

    duplicates_removed = cleaning.get("duplicates_removed", 0)
    if duplicates_removed > 0:
        insights.append(f"{duplicates_removed:,} duplicate rows were removed.")

    dropped_cols = cleaning.get("duplicate_columns_dropped", [])
    if dropped_cols:
        insights.append(f"Identical duplicate columns were removed: {', '.join(dropped_cols)}.")

    remaining_missing = int(df.isna().sum().sum())
    if remaining_missing == 0:
        insights.append("The dataset contains no missing values.")
    else:
        insights.append(f"The dataset contains {remaining_missing:,} remaining missing values.")

    insights.append(f"The dataset has {len(df):,} rows and {len(df.columns)} columns.")

    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    if len(categorical_cols) > 0:
        col = categorical_cols[0]
        top_val = df[col].dropna().mode()
        if not top_val.empty:
            insights.append(f"The most frequent value in '{col}' is '{top_val.iloc[0]}'.")

    return insights[:15]


def analyze_dataset(df, cleaning=None):
    original_df = df.copy()
    if cleaning is None:
        cleaned_df, cleaning = clean_dataset(original_df)
    else:
        cleaned_df = original_df.copy()

    total_missing = int(original_df.isna().sum().sum())
    total_duplicates = int(original_df.duplicated().sum())

    if total_missing == 0 and total_duplicates == 0:
        quality_status = "Excellent"
    elif total_missing < max(len(original_df) * 0.05, 1) and total_duplicates < max(len(original_df) * 0.05, 1):
        quality_status = "Good"
    else:
        quality_status = "Needs Attention"

    result = {
        "rows": int(len(original_df)),
        "columns": int(len(original_df.columns)),
        "missing_values": total_missing,
        "duplicates": total_duplicates,
        "data_quality": {
            "duplicate_rows": total_duplicates,
            "missing_values": total_missing,
            "status": quality_status},
        "cleaning": cleaning,
        "column_types": detect_column_types(original_df),
        "columns_info": column_details(original_df),
        "preview": make_preview(original_df),
        "kpis": generate_kpis(original_df),
        "categories": generate_categories(original_df),
        "distribution": generate_distribution(original_df),
        "insights": generate_insights(cleaned_df, cleaning)}

    return make_json_safe(result)


def create_chart(df, x_column, y_column=None, chart_type="bar"):
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' does not exist.")
    if y_column and y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' does not exist.")

    if chart_type == "pie":
        counts = df[x_column].fillna("Unknown").astype(str).value_counts().head(15)
        return make_json_safe({
            "type": "pie",
            "labels": counts.index.tolist(),
            "values": counts.values.tolist(),
            "x": x_column,
            "y": None})

    if chart_type == "scatter":
        if not y_column:
            raise ValueError("Scatter chart requires a Y-axis.")
        temp = df[[x_column, y_column]].dropna().head(3000)
        return make_json_safe({
            "type": "scatter",
            "labels": temp[x_column].tolist(),
            "values": temp[y_column].tolist(),
            "x": x_column,
            "y": y_column})

    if not y_column:
        raise ValueError("Please select a Y-axis column.")

    temp = df[[x_column, y_column]].dropna()
    grouped = temp.groupby(x_column)[y_column].mean().head(20)

    return make_json_safe({
        "type": "bar",
        "labels": [str(x) for x in grouped.index],
        "values": grouped.values.tolist(),
        "x": x_column,
        "y": y_column})