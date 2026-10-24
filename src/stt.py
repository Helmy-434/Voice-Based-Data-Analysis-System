import os
from faster_whisper import WhisperModel

class SpeechToText:
    def __init__(self, model_size: str = "base", device: str = "cpu"):  # Changed parameter name to model_size
        self.whisper_size = model_size
        self.device = device
        
        # Lazy initialization
        self._whisper = None
        self._qwencleo = None

    @property
    def whisper(self):
        if self._whisper is None:
            self._whisper = WhisperModel(self.whisper_size, device=self.device, compute_type="int8")
        return self._whisper

    @property
    def qwencleo(self):
        if self._qwencleo is None:
            from qwencleo_asr import QwenCleoASR
            self._qwencleo = QwenCleoASR.from_pretrained("mohammedaly22/QwenCleo-ASR", device=self.device)
        return self._qwencleo

    def transcribe(self, audio_file_path: str, language: str = "auto") -> dict:
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found at {audio_file_path}")

        lang = (language or "auto").lower()

        # 1. Explicit Arabic -> QwenCleo-ASR
        if lang == "ar":
            res = self.qwencleo.transcribe(audio_file_path, language="Arabic")
            text = res.text if hasattr(res, "text") else str(res)
            return {
                "text": text.strip(),
                "detected_language": "ar",
                "engine_used": "QwenCleo-ASR"
            }

        # 2. Explicit English -> Faster-Whisper
        elif lang == "en":
            segments, info = self.whisper.transcribe(audio_file_path, language="en", beam_size=5, vad_filter=True)
            text = " ".join([segment.text for segment in segments]).strip()
            return {
                "text": text,
                "detected_language": "en",
                "engine_used": "Faster-Whisper"
            }

        # 3. Auto-Detect Mode
        else:
            segments, info = self.whisper.transcribe(audio_file_path, beam_size=5, vad_filter=True)
            detected_lang = info.language

            if detected_lang == "ar":
                res = self.qwencleo.transcribe(audio_file_path, language="Arabic")
                text = res.text if hasattr(res, "text") else str(res)
                return {
                    "text": text.strip(),
                    "detected_language": "ar",
                    "engine_used": "QwenCleo-ASR (Auto-Routed)"
                }

            text = " ".join([segment.text for segment in segments]).strip()
            return {
                "text": text,
                "detected_language": detected_lang,
                "engine_used": "Faster-Whisper"
            }