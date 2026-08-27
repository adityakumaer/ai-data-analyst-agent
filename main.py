import io
import os
import uuid
import tempfile
from pathlib import Path
from io import BytesIO
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from mangum import Mangum
from ai_agent import ask_gemini
from analyzer import (
    load_dataset, clean_dataset, analyze_dataset, 
    create_chart, make_json_safe, make_preview, 
    column_details, detect_column_types, generate_kpis
)
from db import (
    upload_file_to_supabase, download_file_from_supabase, 
    save_metadata, get_metadata, delete_metadata_and_file
)

app = FastAPI(title="DataLens AI", description="AI Data Analyst Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "frontend"

class ChatRequest(BaseModel):
    question: str

class ChartRequest(BaseModel):
    x_column: str
    y_column: str | None = None
    chart_type: str = "bar"

@app.get("/", response_class=HTMLResponse)
def root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text(encoding="utf-8")
    return "<h3>DataLens AI Dashboard UI not found.</h3>"

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Please upload a file.")

    original_filename = file.filename
    filename = original_filename.lower()
    allowed_extensions = (".csv", ".xlsx", ".xls")
    if not filename.endswith(allowed_extensions):
        raise HTTPException(status_code=400, detail="Only CSV, XLSX and XLS files are supported.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    dataset_id = str(uuid.uuid4())
    
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(content)
        temp_path = temp.name

    try:
        original_df = load_dataset(temp_path)
        clean_df, cleaning = clean_dataset(original_df)
        analysis = analyze_dataset(original_df, cleaning=cleaning)

        storage_path = upload_file_to_supabase(dataset_id, content, original_filename)

        save_metadata(dataset_id, original_filename, analysis, cleaning, storage_path)

        analysis["dataset_id"] = dataset_id
        analysis["dataset"] = {
            "filename": original_filename,
            "original_rows": int(len(original_df)),
            "original_columns": int(len(original_df.columns)),
            "cleaned_rows": int(len(clean_df)),
            "cleaned_columns": int(len(clean_df.columns))
        }
        analysis["original_preview"] = make_preview(original_df, rows=15)
        analysis["cleaning"] = make_json_safe(cleaning)

        return make_json_safe(analysis)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def load_df_from_supabase_meta(dataset_id: str):
    meta = get_metadata(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    
    file_bytes = download_file_from_supabase(meta["storage_path"])
    suffix = os.path.splitext(meta["filename"])[1]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(file_bytes)
        temp_path = temp.name
        
    try:
        original_df = load_dataset(temp_path)
        clean_df, _ = clean_dataset(original_df)
        return original_df, clean_df, meta
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/dataset/{dataset_id}")
def get_dataset(dataset_id: str):
    meta = get_metadata(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return make_json_safe(meta["analysis"])

@app.get("/preview/{dataset_id}")
def get_original_preview(dataset_id: str):
    original_df, _, _ = load_df_from_supabase_meta(dataset_id)
    return make_json_safe({
        "rows": int(len(original_df)),
        "columns": int(len(original_df.columns)),
        "data": make_preview(original_df, rows=50),
        "column_info": column_details(original_df),
        "kpis": generate_kpis(original_df)
    })

@app.get("/clean-preview/{dataset_id}")
def get_clean_preview(dataset_id: str):
    _, clean_df, meta = load_df_from_supabase_meta(dataset_id)
    return make_json_safe({
        "rows": int(len(clean_df)),
        "columns": int(len(clean_df.columns)),
        "data": make_preview(clean_df, rows=50),
        "column_info": column_details(clean_df),
        "kpis": generate_kpis(clean_df),
        "cleaning": meta["cleaning"]
    })

@app.post("/chat/{dataset_id}")
def chat(dataset_id: str, request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    original_df, clean_df, meta = load_df_from_supabase_meta(dataset_id)

    try:
        answer = ask_gemini(
            question=question,
            analysis=meta["analysis"],
            original_df=original_df,
            clean_df=clean_df)
        return {"answer": answer}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

@app.post("/chart/{dataset_id}")
def generate_custom_chart(dataset_id: str, request: ChartRequest):
    _, clean_df, _ = load_df_from_supabase_meta(dataset_id)
    try:
        chart_data = create_chart(
            df=clean_df,
            x_column=request.x_column,
            y_column=request.y_column,
            chart_type=request.chart_type
        )
        return chart_data
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))

@app.delete("/dataset/{dataset_id}")
def delete_dataset(dataset_id: str):
    meta = get_metadata(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    delete_metadata_and_file(dataset_id, meta["storage_path"])
    return {"message": "Dataset deleted successfully."}

handler = Mangum(app)