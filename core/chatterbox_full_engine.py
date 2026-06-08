from pathlib import Path
from typing import Optional

import numpy as np
import torch

from core.tts_engine import BaseTTSEngine, TTSOutput, TTSFactory, VoiceProfile


@TTSFactory.register("chatterbox_full")
class ChatterboxFullEngine(BaseTTSEngine):

    def __init__(self):
        self._model = None
        self._device: str = "cuda"
        self._sr: int = 44100
        self.exaggeration = 0.5
        self.cfg_weight = 0.5

    def initialize(self) -> None:
        from chatterbox.tts import ChatterboxTTS
        self._model = ChatterboxTTS.from_pretrained(device=self._device)
        self._sr = self._model.sr

    @property
    def sample_rate(self) -> int:
        return self._sr

    def generate(self, text: str, voice: Optional[VoiceProfile] = None, speed: float = 1.0) -> TTSOutput:
        if self._model is None:
            raise RuntimeError("Engine not initialized")

        ref_path = None
        if voice is not None and voice.reference_path is not None:
            ref_path = Path(voice.reference_path)
            if not ref_path.exists():
                ref_path = None

        if ref_path is None:
            ref_path = self._default_ref_path()

        if ref_path is None or not Path(str(ref_path)).exists():
            raise FileNotFoundError(
                "No reference voice clip found. Please import a voice clip first."
            )

        wav_tensor = self._model.generate(
            text, audio_prompt_path=str(ref_path),
            exaggeration=self.exaggeration,
            cfg_weight=self.cfg_weight,
        )
        audio = wav_tensor.squeeze().cpu().numpy()
        duration = len(audio) / self._sr
        return TTSOutput(audio=audio, sample_rate=self._sr, duration_seconds=duration)

    def list_voices(self) -> list[VoiceProfile]:
        presets_dir = Path(__file__).parent.parent / "voices" / "presets"
        voices: list[VoiceProfile] = []
        if presets_dir.exists():
            for f in sorted(presets_dir.glob("*.wav")):
                voices.append(VoiceProfile(
                    name=f.stem, reference_path=str(f), is_builtin=True,
                ))
        custom_dir = Path(__file__).parent.parent / "voices" / "custom"
        if custom_dir.exists():
            for f in sorted(custom_dir.glob("*.wav")):
                voices.append(VoiceProfile(
                    name=f.stem, reference_path=str(f), is_builtin=False,
                ))
        return voices

    def _default_ref_path(self) -> Optional[Path]:
        voices = self.list_voices()
        if voices:
            return Path(voices[0].reference_path)
        return None

    def unload(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
