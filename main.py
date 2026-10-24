import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.sql_engine import SQLiteEngine
from dotenv import load_dotenv


load_dotenv()



app = FastAPI(title="Voice Data Analysis API")

# Keep the loaded table available between upload and analysis requests.
database = SQLiteEngine()
dataset_info: dict[str, Any] | None = None
stt_engine = None
sql_generator = None


def get_stt_engine():
    global stt_engine
    if stt_engine is None:
        from src.stt import SpeechToText

        stt_engine = SpeechToText(model_size=os.getenv("WHISPER_MODEL", "base"))
    return stt_engine


def get_sql_generator():
    global sql_generator
    if sql_generator is None:
        from src.sql_generator import SQLGenerator

        sql_generator = SQLGenerator()
    return sql_generator


@app.get("/")
async def health_check():
    return {"status": "ok", "service": "voice-data-analysis"}


@app.post("/upload-dataset")
async def upload_dataset(dataset_file: UploadFile = File(...)):
    global dataset_info
    filename = Path(dataset_file.filename or "").name
    if Path(filename).suffix.lower() not in {".csv", ".json", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Upload a CSV, JSON, or Excel dataset.")
    try:
        contents = await dataset_file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded dataset is empty.")
        dataset_info = database.load_dataset(contents, filename)
        return {"filename": filename, **dataset_info}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read dataset: {exc}") from exc
    finally:
        await dataset_file.close()


@app.post("/analyze-voice")
async def analyze_voice(
    audio_file: UploadFile = File(...),
    language: str = Form("auto"),
):
    if dataset_info is None:
        raise HTTPException(status_code=409, detail="Upload a dataset before recording a question.")
    if language not in {"auto", "ar", "en"}:
        raise HTTPException(status_code=400, detail="Language must be auto, ar, or en.")

    suffix = Path(audio_file.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".webm", ".ogg", ".mp4"}:
        suffix = ".audio"
    audio_bytes = await audio_file.read()
    await audio_file.close()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="The audio recording is empty.")

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio:
            audio.write(audio_bytes)
            audio_path = audio.name
        transcription = get_stt_engine().transcribe(audio_path, language=language)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Speech transcription failed: {exc}") from exc
    finally:
        if "audio_path" in locals():
            Path(audio_path).unlink(missing_ok=True)

    question = transcription.get("text", "").strip()
    if not question:
        raise HTTPException(status_code=422, detail="No speech was recognized. Please try again.")

    try:
        sql = get_sql_generator().generate_sql(
            question,
            dataset_info["table_name"],
            dataset_info["schema_info"],
        )

        connection = database.conn
        cursor = connection.execute(sql)

        # Check if the query returns a result set or modifies data 
        if cursor.description is None:
            connection.commit() # Save changes to the database
            return {
                "transcription": question,
                "language_used": transcription.get("detected_language"),
                "language_probability": transcription.get("language_probability"),
                "sql": sql,
                "status": "success",
                "message": "Query executed successfully. Database modified.",
                "rows_affected": cursor.rowcount
            }
        else:
            columns = [column[0] for column in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchmany(500)]
            return {
                "transcription": question,
                "language_used": transcription.get("detected_language"),
                "language_probability": transcription.get("language_probability"),
                "sql": sql,
                "columns": columns,
                "rows": rows,
                "truncated": len(rows) == 500,
            }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
