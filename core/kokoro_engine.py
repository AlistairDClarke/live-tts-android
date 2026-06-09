import array
import os
import threading
import time
from pathlib import Path
from typing import Optional

from core.tts_engine import BaseTTSEngine, TTSOutput, TTSFactory, VoiceProfile

SHERPA_PACKAGES = [
    "com.k2fsa.sherpa.onnx.tts.engine",
    "org.sherpa.onnx.tts.engine",
]

KOKORO_VOICES = [
    "af_heart", "af_bella", "af_nicole", "af_aoede", "af_kore", "af_sarah",
    "af_nova", "af_sky", "af_alloy", "af_jessica", "af_river", "af_michaela",
    "af_cove", "af_brooke", "af_iris", "af_catherine", "am_adam", "am_michael",
    "am_echo", "am_eric", "am_liam", "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_lily", "bf_george", "bf_daniel", "bf_isabella",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora", "ef_emma",
    "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta",
    "hm_omega", "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
    "jm_kumo",
    "pf_dora", "pf_emma",
    "pm_alex", "pm_santa",
    "sf_bella", "sf_brooke",
    "tf_alpha", "tf_beta",
    "tm_omega", "tm_psi",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxia", "zm_yunyang",
]


@TTSFactory.register("kokoro")
class KokoroEngine(BaseTTSEngine):

    def __init__(self):
        self._tts = None
        self._ready = False
        self._ready_event = threading.Event()
        self._sr: int = 24000
        self._current_voice: str = "af_heart"
        self._engine_package: Optional[str] = None
        self._voices_cache: list[VoiceProfile] = []
        self._temp_dir: Optional[str] = None
        self._pending_error: Optional[str] = None
        self._synth_done = threading.Event()
        self._synth_file: Optional[str] = None

    @property
    def sample_rate(self) -> int:
        return self._sr

    def initialize(self) -> None:
        try:
            from jnius import autoclass, cast
            from android.config import ACTIVITY_CLASS_NAME
        except ImportError:
            raise RuntimeError(
                "Android TTS requires pyjnius. "
                "Install the Sherpa-ONNX TTS Engine APK from "
                "https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine.html"
            )

        self._PythonActivity = autoclass(ACTIVITY_CLASS_NAME)
        self._TextToSpeech = autoclass("android.speech.tts.TextToSpeech")
        self._Bundle = autoclass("android.os.Bundle")
        self._File = autoclass("java.io.File")
        self._Locale = autoclass("java.util.Locale")
        self._Voice = autoclass("android.speech.tts.Voice")

        import tempfile
        self._temp_dir = tempfile.mkdtemp(prefix="tts_")

        self._ready_event.clear()
        listener = self._TTSInitListener(self)
        self._tts = self._TextToSpeech(
            self._PythonActivity.mActivity, listener
        )

        if not self._ready_event.wait(timeout=15):
            self._tts.shutdown()
            raise RuntimeError("TTS engine initialization timed out")

        if self._pending_error:
            err = self._pending_error
            self._pending_error = None
            raise RuntimeError(f"TTS init failed: {err}")

        self._detect_sherpa_engine()
        self._cache_voices()

    class _TTSInitListener:
        def __init__(self, engine):
            self._engine = engine

        def onInit(self, status):
            if status == 0:  # SUCCESS
                self._engine._ready = True
                self._engine._ready_event.set()
            else:
                self._engine._pending_error = f"Init status: {status}"
                self._engine._ready_event.set()

    def _detect_sherpa_engine(self):
        engines = self._tts.getEngines()
        it = engines.iterator()
        while it.hasNext():
            entry = it.next()
            pkg = entry.getKey()
            for prefix in SHERPA_PACKAGES:
                if pkg.startswith(prefix):
                    self._engine_package = pkg
                    self._tts.setEngineByPackageName(pkg)
                    return

        raise RuntimeError(
            "Sherpa-ONNX TTS Engine not found. "
            "Please install it from "
            "https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine.html"
        )

    def _cache_voices(self):
        self._voices_cache = []
        voices = self._tts.getVoices()
        it = voices.iterator()
        while it.hasNext():
            voice = it.next()
            name = voice.getName()
            self._voices_cache.append(VoiceProfile(
                name=name,
                is_builtin=True,
                metadata={
                    "kokoro_voice": name,
                    "locale": str(voice.getLocale()),
                },
            ))

        if not self._voices_cache:
            for v in KOKORO_VOICES:
                self._voices_cache.append(VoiceProfile(
                    name=v,
                    is_builtin=True,
                    metadata={"kokoro_voice": v},
                ))

    def generate(
        self,
        text: str,
        voice: Optional[VoiceProfile] = None,
        speed: float = 1.0,
    ) -> TTSOutput:
        if self._tts is None or not self._ready:
            raise RuntimeError("Engine not initialized")

        voice_name = self._current_voice
        if voice is not None and voice.metadata.get("kokoro_voice"):
            voice_name = voice.metadata["kokoro_voice"]

        self._tts.setSpeechRate(speed)

        import uuid
        filename = os.path.join(
            self._temp_dir, f"tts_{uuid.uuid4().hex}.wav"
        )

        self._synth_done.clear()
        self._synth_file = None

        result = self._tts.synthesizeToFile(
            text, None, self._File(filename), "tts_1"
        )

        time.sleep(0.3 + len(text) * 0.03)

        if result != 0:  # SUCCESS
            raise RuntimeError(f"TTS synthesis failed with code {result}")

        if not os.path.exists(filename):
            raise RuntimeError("TTS output file not created")

        import wave
        with wave.open(filename, "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)

        if sampwidth == 2:
            arr = array.array("h")
            arr.frombytes(raw)
            max_val = 32767.0
        elif sampwidth == 4:
            arr = array.array("i")
            arr.frombytes(raw)
            max_val = 2147483647.0

        if nchannels > 1:
            mono = array.array("f", [0.0]) * (len(arr) // nchannels)
            for i in range(len(mono)):
                total = sum(arr[i * nchannels + ch] for ch in range(nchannels))
                mono[i] = (total / nchannels) / max_val
            audio = list(mono)
        else:
            audio = [s / max_val for s in arr]

        duration = len(audio) / framerate

        try:
            os.unlink(filename)
        except Exception:
            pass

        return TTSOutput(
            audio=audio,
            sample_rate=framerate,
            duration_seconds=duration,
        )

    def list_voices(self) -> list[VoiceProfile]:
        if self._voices_cache:
            return self._voices_cache
        return [
            VoiceProfile(name=v, is_builtin=True, metadata={"kokoro_voice": v})
            for v in KOKORO_VOICES
        ]

    def unload(self) -> None:
        if self._tts is not None:
            try:
                self._tts.shutdown()
            except Exception:
                pass
            self._tts = None
            self._ready = False
        if self._temp_dir:
            import shutil
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass
