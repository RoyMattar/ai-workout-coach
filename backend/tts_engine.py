"""
Text-to-Speech Engine

Provides voice coaching feedback using pre-trained TTS models.
Supports multiple providers and voice selection for coach personas.

Primary: OpenAI TTS (high-quality neural voices with selection)
Secondary: gTTS (Google WaveNet, single voice)
Fallback: pyttsx3 (offline system TTS)
"""
import hashlib
import io
import logging
import base64
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Valid OpenAI TTS voices
OPENAI_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}


class TTSEngine:
    """
    Text-to-Speech engine with multiple providers and voice selection.

    Provider priority:
    1. OpenAI TTS (if API key available) — best quality, voice selection
    2. gTTS (if installed) — good quality, no voice selection
    3. pyttsx3 (if installed) — offline fallback
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        default_voice: str = "nova",
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.default_voice = default_voice
        self.use_cache = use_cache
        self._cache: dict[str, bytes] = {}

        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="tts_cache_"))

        # Initialize providers
        self._openai_client = None
        if openai_api_key is None:
            from .config import get_settings
            openai_api_key = get_settings().openai_api_key

        if openai_api_key:
            try:
                import openai
                self._openai_client = openai.OpenAI(api_key=openai_api_key)
                logger.info("TTS: OpenAI TTS available (6 voices)")
            except Exception as e:
                logger.warning(f"OpenAI TTS init failed: {e}")

        self._gtts_available = self._check_gtts()
        self._pyttsx3_available = self._check_pyttsx3()

        if not self._openai_client and self._gtts_available:
            logger.info("TTS: gTTS available as primary")
        if self._pyttsx3_available:
            logger.info("TTS: pyttsx3 available as fallback")

    @staticmethod
    def _check_gtts() -> bool:
        try:
            import gtts  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_pyttsx3() -> bool:
        try:
            import pyttsx3  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        return self._openai_client is not None or self._gtts_available or self._pyttsx3_available

    @property
    def engine_name(self) -> str:
        if self._openai_client:
            return "OpenAI TTS"
        elif self._gtts_available:
            return "gTTS (Google WaveNet)"
        elif self._pyttsx3_available:
            return "pyttsx3 (offline)"
        return "none"

    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Optional[bytes]:
        """
        Convert text to speech audio (MP3 bytes).

        Args:
            text: The text to convert to speech.
            voice: OpenAI voice ID (alloy, echo, fable, onyx, nova, shimmer).
                   Only used with OpenAI TTS provider.
            speed: Speech speed multiplier (0.25 to 4.0). OpenAI TTS only.
        """
        if not text or not text.strip():
            return None

        if not self.is_available:
            return None

        voice = voice or self.default_voice
        cache_key = self._cache_key(text, voice)

        if self.use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        audio = None

        # Try OpenAI TTS first (best quality, voice selection)
        if self._openai_client:
            audio = self._synthesize_openai(text, voice, speed)

        # Fallback to gTTS
        if audio is None and self._gtts_available:
            audio = self._synthesize_gtts(text)

        # Fallback to pyttsx3
        if audio is None and self._pyttsx3_available:
            audio = self._synthesize_pyttsx3(text)

        if audio and self.use_cache:
            self._cache[cache_key] = audio

        return audio

    def synthesize_base64(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> Optional[str]:
        """Convert text to speech and return as base64-encoded string."""
        audio = self.synthesize(text, voice, speed)
        if audio:
            return base64.b64encode(audio).decode("utf-8")
        return None

    def _synthesize_openai(
        self,
        text: str,
        voice: str = "nova",
        speed: float = 1.0,
    ) -> Optional[bytes]:
        """Synthesize speech using OpenAI TTS API."""
        try:
            if voice not in OPENAI_VOICES:
                voice = "nova"

            speed = max(0.25, min(4.0, speed))

            response = self._openai_client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
                speed=speed,
                response_format="mp3",
            )

            audio_bytes = response.content
            logger.debug(f"OpenAI TTS: {len(audio_bytes)} bytes, voice={voice}")
            return audio_bytes

        except Exception as e:
            logger.error(f"OpenAI TTS failed: {e}")
            return None

    def _synthesize_gtts(self, text: str) -> Optional[bytes]:
        """Synthesize speech using Google Text-to-Speech."""
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang="en", slow=False)
            buffer = io.BytesIO()
            tts.write_to_fp(buffer)
            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            logger.error(f"gTTS failed: {e}")
            return None

    def _synthesize_pyttsx3(self, text: str) -> Optional[bytes]:
        """Synthesize speech using pyttsx3 (offline fallback)."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            engine.setProperty("volume", 0.9)
            temp_path = self.cache_dir / f"tts_{self._cache_key(text)}.mp3"
            engine.save_to_file(text, str(temp_path))
            engine.runAndWait()
            if temp_path.exists():
                audio_bytes = temp_path.read_bytes()
                temp_path.unlink()
                return audio_bytes
            return None
        except Exception as e:
            logger.error(f"pyttsx3 failed: {e}")
            return None

    @staticmethod
    def _cache_key(text: str, voice: str = "default") -> str:
        """Generate a cache key from text and voice."""
        return hashlib.md5(f"{voice}:{text.strip().lower()}".encode()).hexdigest()

    def clear_cache(self):
        self._cache.clear()

    def get_info(self) -> dict:
        return {
            "engine": self.engine_name,
            "openai_available": self._openai_client is not None,
            "gtts_available": self._gtts_available,
            "pyttsx3_available": self._pyttsx3_available,
            "voices": list(OPENAI_VOICES) if self._openai_client else [],
            "cache_size": len(self._cache),
        }
