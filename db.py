import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "datasets"

_supabase_client = None

def get_supabase_client() -> Client:
    global _supabase_client
    if _supabase_client is None:
        url = os.environ.get("SUPABASE_URL") or SUPABASE_URL
        key = os.environ.get("SUPABASE_KEY") or SUPABASE_KEY
        if not url or not key:
            raise ValueError("Supabase URL or API key is missing from environment variables.")
        _supabase_client = create_client(url, key)
    return _supabase_client

def upload_file_to_supabase(dataset_id: str, file_bytes: bytes, filename: str) -> str:
    client = get_supabase_client()
    path = f"{dataset_id}/{filename}"
    client.storage.from_(BUCKET_NAME.lower()).upload(
        path=path,
        file=file_bytes,
        file_options={"content-type": "application/octet-stream", "upsert": "true"})
    return path

def download_file_from_supabase(path: str) -> bytes:
    client = get_supabase_client()
    response = client.storage.from_(BUCKET_NAME.lower()).download(path)
    return response

def save_metadata(dataset_id: str, filename: str, analysis: dict, cleaning: dict, storage_path: str):
    client = get_supabase_client()
    data = {
        "dataset_id": dataset_id,
        "filename": filename,
        "analysis": analysis,
        "cleaning": cleaning,
        "storage_path": storage_path}
    response = client.table("datasets_meta").upsert(data).execute()
    return response

def get_metadata(dataset_id: str) -> dict | None:
    client = get_supabase_client()
    response = client.table("datasets_meta").select("*").eq("dataset_id", dataset_id).execute()
    if response.data and len(response.data) > 0:
        return response.data[0]
    return None

def delete_metadata_and_file(dataset_id: str, storage_path: str):
    try:
        client = get_supabase_client()
        client.storage.from_(BUCKET_NAME.lower()).remove([storage_path])
        client.table("datasets_meta").delete().eq("dataset_id", dataset_id).execute()
    except Exception:
        pass