import os
import json
import pandas as pd
from dotenv import load_dotenv
from google import genai


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

def ask_gemini(
    question: str,
    analysis: dict,
    original_df: pd.DataFrame,
    clean_df: pd.DataFrame):
    
    if not GEMINI_API_KEY or client is None:
        return "Gemini API key is not configured or client failed to initialize."

    cleaning_info = analysis.get("cleaning", {})

    orig_rows, orig_cols = original_df.shape
    clean_rows, clean_cols = clean_df.shape
    rows_removed = cleaning_info.get("rows_removed", orig_rows - clean_rows)
    cols_removed = cleaning_info.get("columns_removed", orig_cols - clean_cols)
    dropped_cols = cleaning_info.get("duplicate_columns_dropped", [])
    duplicates_removed = cleaning_info.get("duplicates_removed", 0)

    gender_changes_note = ""
    if "Patient Gender" in original_df.columns and "Patient Gender" in clean_df.columns:
        orig_genders = original_df["Patient Gender"].value_counts().to_dict()
        clean_genders = clean_df["Patient Gender"].value_counts().to_dict()
        gender_changes_note = f"""
        - Patient Gender Column: 
          * Original values found: {orig_genders}
          * Cleaned values standardized to: {clean_genders}
          * Explanation: Shorthand codes ('M', 'F') and case variations were unified into standardized categories ('Male', 'Female').
        """

    context_summary = {
        "original_shape": {"rows": orig_rows, "columns": orig_cols},
        "cleaned_shape": {"rows": clean_rows, "columns": clean_cols},
        "rows_removed": rows_removed,
        "columns_removed": cols_removed,
        "duplicate_columns_dropped": dropped_cols,
        "duplicate_rows_removed": duplicates_removed,
        "cleaning_actions": [
            "Harmonized unstandardized text variations (e.g., 'M', 'Male', 'F', 'Female' unified into standard format).",
            "Dropped 100% duplicate/redundant columns (e.g., duplicate flag columns).",
            "Removed completely empty rows and duplicate entries.",
            "Handled missing values conservatively."]}

    sample_cleaned = clean_df.head(50).to_dict(orient="records")

    prompt = f"""
You are an expert AI Data Analyst and Data Cleaning Assistant for 'DataLens AI'. 
You have direct access to both the **ORIGINAL raw dataset** and the **CLEANED dataset**.

When the user asks:
- "How was my data cleaned?" -> Detail all cleaning steps performed (dropping duplicate columns/rows, standardizing categories, handling missing values).
- "What changes were made to the Patient Gender column?" -> Explain how variations like 'M', 'Male', 'F', and 'Female' were mapped and unified.
- "What is the difference between my original file and the cleaned file?" -> Provide exact row/column counts, explain why rows or duplicate columns were removed.

==================================================
DATASET CLEANING REPORT & COMPARISON
==================================================
{json.dumps(context_summary, indent=2, default=str)}

{gender_changes_note}

==================================================
CLEANED DATASET SAMPLE (Top 50 rows)
==================================================
{json.dumps(sample_cleaned, indent=2, default=str)}

==================================================
USER QUESTION
==================================================
{question}

==================================================
RESPONSE GUIDELINES
==================================================
1. Be clear, professional, and precise.
2. Use exact numbers and statistics when comparing original vs. cleaned files.
3. Use Markdown headings, bullet points, and tables where appropriate.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt)
        if not response.text:
            return "Gemini returned an empty response."
        return response.text
    except Exception as error:
        print("GEMINI ERROR:", str(error))
        return f"I couldn't process your question right now. Error: {str(error)}"