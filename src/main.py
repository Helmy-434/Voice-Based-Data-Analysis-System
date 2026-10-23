from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from src.stt import SpeechToText

app = FastAPI()
stt_engine = SpeechToText(model_size="base")
app.add_middleware()

@app.get("/")
async def root():
    return {"Hello": "World"}


@app.post("/analyze-voice")
async def analyze_voice(
        audio_file: UploadFile = File(...),
        language: str = Form("auto")  # Accepts 'ar', 'en', or 'auto'
):
    saved_audio_path = f"./uploads/{audio_file.filename}"

    transcription = stt_engine.transcribe(
        audio_file_path=saved_audio_path,
        language=language
    )

    return {
        "text": transcription["text"],
        "language_used": transcription["detected_language"]
    }