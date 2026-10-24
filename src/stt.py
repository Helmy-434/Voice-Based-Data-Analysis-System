import os
from faster_whisper import WhisperModel

class SpeechToText:
    def __init__(self, model_size: str = "base", device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self._model = None

    @property
    def model(self):
        # Lazy loads the model into memory
        if self._model is None:
            self._model = WhisperModel(
                self.model_size, 
                device=self.device, 
                compute_type="int8"
            )
        return self._model

    def transcribe(self, audio_file_path: str, language: str = "auto") -> dict:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found at {audio_file_path}")

        lang = None if language in ["auto", None, ""] else language.lower()

        segments, info = self.model.transcribe(
            audio_file_path,
            language=lang,
            beam_size=5,
            vad_filter=True
        )

        transcription_text = " ".join([segment.text for segment in segments]).strip()

        return {
            "text": transcription_text,
            "detected_language": info.language if lang is None else lang,
            "language_probability": round(info.language_probability, 2)
        }