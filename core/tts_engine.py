from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, List


@dataclass
class TTSOutput:
    audio: List[float]
    sample_rate: int
    duration_seconds: float


@dataclass
class VoiceProfile:
    name: str
    reference_path: Optional[str] = None
    is_builtin: bool = False
    metadata: dict = field(default_factory=dict)


class BaseTTSEngine(ABC):

    @abstractmethod
    def initialize(self) -> None:
        ...

    @abstractmethod
    def generate(self, text: str, voice: Optional[VoiceProfile] = None) -> TTSOutput:
        ...

    @abstractmethod
    def list_voices(self) -> list[VoiceProfile]:
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...


class TTSFactory:
    _engines: dict[str, type[BaseTTSEngine]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(engine_cls: type[BaseTTSEngine]):
            cls._engines[name] = engine_cls
            return engine_cls
        return decorator

    @classmethod
    def create(cls, name: str, device: str = "cuda") -> BaseTTSEngine:
        engine_cls = cls._engines.get(name)
        if engine_cls is None:
            raise ValueError(f"Unknown engine: {name}. Available: {list(cls._engines.keys())}")
        engine = engine_cls()
        engine._device = device
        engine.initialize()
        return engine
