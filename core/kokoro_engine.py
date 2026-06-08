from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.tts_engine import BaseTTSEngine, TTSOutput, TTSFactory, VoiceProfile


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
    "if_sara", "if_sara",
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
        self._pipeline = None
        self._device: str = "cpu"
        self._sr: int = 24000
        self._current_voice: str = "af_heart"

    def initialize(self) -> None:
        from kokoro import KPipeline
        self._pipeline = KPipeline(
            lang_code='a',
            repo_id='hexgrad/Kokoro-82M',
            device=self._device,
        )

        self._pipeline("warmup", voice=self._current_voice, speed=1.0).__next__()

    @property
    def sample_rate(self) -> int:
        return self._sr

    def generate(self, text: str, voice: Optional[VoiceProfile] = None, speed: float = 1.0) -> TTSOutput:
        if self._pipeline is None:
            raise RuntimeError("Engine not initialized")

        voice_name = self._current_voice
        if voice is not None and voice.metadata.get("kokoro_voice"):
            voice_name = voice.metadata["kokoro_voice"]

        audio_chunks: list = []
        generator = self._pipeline(text, voice=voice_name, speed=speed)
        for _, _, audio in generator:
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            elif hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            audio_chunks.append(audio)

        if not audio_chunks:
            audio = self._pipeline(text, voice=voice_name, speed=speed).__next__()[2]
            if hasattr(audio, 'numpy'):
                audio = audio.numpy()
            elif hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()
            audio_chunks = [audio]

        combined = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
        duration = len(combined) / self._sr
        return TTSOutput(audio=combined, sample_rate=self._sr, duration_seconds=duration)

    def list_voices(self) -> list[VoiceProfile]:
        voices: list[VoiceProfile] = []
        for v in KOKORO_VOICES:
            voices.append(VoiceProfile(
                name=v,
                is_builtin=True,
                metadata={"kokoro_voice": v},
            ))
        return voices

    def unload(self) -> None:
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
