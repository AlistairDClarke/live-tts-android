from pathlib import Path
from typing import Optional

import numpy as np

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

KOKORO_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "tts-models/kokoro-multi-lang-v1_0.tar.bz2"
)


@TTSFactory.register("kokoro")
class KokoroEngine(BaseTTSEngine):

    def __init__(self):
        self._tts = None
        self._config = None
        self._device: str = "cpu"
        self._sr: int = 24000
        self._current_voice: str = "af_heart"
        self._model_dir: Optional[Path] = None

    @property
    def sample_rate(self) -> int:
        return self._sr

    def initialize(self) -> None:
        import sherpa_onnx

        self._model_dir = self._ensure_model()

        self._config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(self._model_dir / "model.onnx"),
                    voices=str(self._model_dir / "voices.bin"),
                    tokens=str(self._model_dir / "tokens.txt"),
                    data_dir=str(self._model_dir / "espeak-ng-data"),
                    lexicon=",".join([
                        str(self._model_dir / "lexicon-us-en.txt"),
                        str(self._model_dir / "lexicon-zh.txt"),
                    ]),
                ),
                provider="cpu",
                debug=False,
                num_threads=2,
            ),
            max_num_sentences=1,
        )

        if not self._config.validate():
            raise RuntimeError("Invalid TTS config")

        self._tts = sherpa_onnx.OfflineTts(self._config)
        self._sr = 24000

    def _ensure_model(self) -> Path:
        import os
        import tarfile
        import urllib.request

        model_dir = Path(__file__).parent.parent / "voices" / "kokoro_model"
        model_file = model_dir / "model.onnx"

        if model_file.exists():
            return model_dir

        model_dir.mkdir(parents=True, exist_ok=True)

        archive_path = model_dir / "kokoro-multi-lang-v1_0.tar.bz2"
        if not archive_path.exists():
            urllib.request.urlretrieve(KOKORO_MODEL_URL, archive_path)

        with tarfile.open(archive_path) as tf:
            members = tf.getmembers()
            if members:
                prefix = members[0].name.split("/")[0] + "/"
                for member in members:
                    if member.name.startswith(prefix):
                        member.name = member.name[len(prefix):]
                    if member.name:
                        tf.extract(member, model_dir)

        try:
            os.unlink(archive_path)
        except Exception:
            pass

        return model_dir

    def generate(
        self,
        text: str,
        voice: Optional[VoiceProfile] = None,
        speed: float = 1.0,
    ) -> TTSOutput:
        if self._tts is None:
            raise RuntimeError("Engine not initialized")

        import sherpa_onnx

        voice_name = self._current_voice
        if voice is not None and voice.metadata.get("kokoro_voice"):
            voice_name = voice.metadata["kokoro_voice"]

        sid = self._voice_name_to_sid(voice_name)

        gen_config = sherpa_onnx.GenerationConfig()
        gen_config.sid = sid
        gen_config.speed = speed
        gen_config.silence_scale = 0.2

        result = self._tts.generate(text, gen_config)

        audio = np.array(result.samples, dtype=np.float32)
        duration = len(audio) / result.sample_rate
        return TTSOutput(
            audio=audio,
            sample_rate=result.sample_rate,
            duration_seconds=duration,
        )

    def _voice_name_to_sid(self, name: str) -> int:
        if name in KOKORO_VOICES:
            return KOKORO_VOICES.index(name)
        return 0

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
        self._tts = None
        self._config = None
