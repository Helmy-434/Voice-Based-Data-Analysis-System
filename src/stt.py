import os
from faster_whisper import WhisperModel

class SpeechToText:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_file_path: str,language: str = None) -> dict:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found at {audio_file_path}")

        target_lang = None
        if language and language.lower() in ["ar", "en"]:
            target_lang = language.lower()

        segments, info = self.model.transcribe(
            audio_file_path,
            language=target_lang,
            beam_size=5,
            vad_filter=True
        )
        transcription_text = " ".join([segment.text for segment in segments]).strip()

        return {
            "text": transcription_text,
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 2)
        }