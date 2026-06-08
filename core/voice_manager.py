import json
import shutil
from pathlib import Path
from typing import Optional

from core.tts_engine import VoiceProfile


class VoiceManager:
    _instance: Optional["VoiceManager"] = None

    def __init__(self):
        self._data_dir = Path(__file__).parent.parent
        self._presets_dir = self._data_dir / "voices" / "presets"
        self._custom_dir = self._data_dir / "voices" / "custom"
        self._profiles_file = self._data_dir / "voices" / "profiles.json"
        self._profiles: dict[str, VoiceProfile] = {}
        self._ensure_dirs()
        self._load()

    @classmethod
    def instance(cls) -> "VoiceManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def custom_dir(self) -> Path:
        return self._custom_dir

    @property
    def presets_dir(self) -> Path:
        return self._presets_dir

    def add_custom_voice(self, name: str, reference_path: str) -> VoiceProfile:
        ref_path = Path(reference_path)
        dest = self._custom_dir / f"{name}.wav"
        if ref_path != dest:
            shutil.copy2(ref_path, dest)
        profile = VoiceProfile(
            name=name,
            reference_path=str(dest),
            is_builtin=False,
        )
        self._profiles[name] = profile
        self._save()
        return profile

    def remove_voice(self, name: str):
        profile = self._profiles.pop(name, None)
        if profile and profile.reference_path:
            path = Path(profile.reference_path)
            if path.exists() and path.parent == self._custom_dir:
                path.unlink(missing_ok=True)
        self._save()

    def get_voice(self, name: str) -> Optional[VoiceProfile]:
        return self._profiles.get(name)

    def list_custom_voices(self) -> list[VoiceProfile]:
        return [p for p in self._profiles.values() if not p.is_builtin]

    def list_all_voices(self) -> list[VoiceProfile]:
        return list(self._profiles.values())

    def _save(self):
        data = {}
        for name, profile in self._profiles.items():
            data[name] = {
                "name": profile.name,
                "reference_path": profile.reference_path,
                "is_builtin": profile.is_builtin,
                "metadata": profile.metadata,
            }
        self._profiles_file.write_text(json.dumps(data, indent=2))

    def _load(self):
        if not self._profiles_file.exists():
            self._scan_dirs()
            return
        data = json.loads(self._profiles_file.read_text())
        self._profiles = {}
        for name, d in data.items():
            self._profiles[name] = VoiceProfile(
                name=d.get("name", name),
                reference_path=d.get("reference_path"),
                is_builtin=d.get("is_builtin", False),
                metadata=d.get("metadata", {}),
            )
        self._scan_dirs()

    def _scan_dirs(self):
        if self._presets_dir.exists():
            for f in sorted(self._presets_dir.glob("*.wav")):
                name = f.stem
                if name not in self._profiles:
                    self._profiles[name] = VoiceProfile(
                        name=name,
                        reference_path=str(f),
                        is_builtin=True,
                    )
        if self._custom_dir.exists():
            for f in sorted(self._custom_dir.glob("*.wav")):
                name = f.stem
                if name not in self._profiles:
                    self._profiles[name] = VoiceProfile(
                        name=name,
                        reference_path=str(f),
                        is_builtin=False,
                    )
        self._save()

    def _ensure_dirs(self):
        self._presets_dir.mkdir(parents=True, exist_ok=True)
        self._custom_dir.mkdir(parents=True, exist_ok=True)
