import array
import os
import sys
import threading
import tempfile
import time
import wave

# Try Android native audio, fall back to Kivy Sound
try:
    from jnius import autoclass
    MediaPlayer = autoclass("android.media.MediaPlayer")
    AudioManager = autoclass("android.media.AudioManager")
    _ANDROID = True
except Exception:
    MediaPlayer = None
    _ANDROID = False


class AudioService:
    """Single-threaded audio service. All audio ops happen here."""

    def __init__(self):
        self._current_player = None
        self._lock = threading.Lock()
        self._dir = tempfile.mkdtemp(prefix="tts_audio_")

    def play(self, audio: list, sample_rate: int):
        path = self._write_wav(audio, sample_rate)
        self._play_file(path)
        try:
            os.unlink(path)
        except Exception:
            pass

    def stop(self):
        with self._lock:
            if self._current_player is not None:
                try:
                    self._current_player.stop()
                    self._current_player.release()
                except Exception:
                    pass
                self._current_player = None

    def _write_wav(self, audio: list, sample_rate: int) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav", dir=self._dir)
        d16 = array.array("h", [max(-32768, min(32767, int(s * 32767))) for s in audio])
        with os.fdopen(fd, "wb") as f:
            with wave.open(f, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(d16.tobytes())
        return path

    def _play_file(self, path: str):
        if _ANDROID and MediaPlayer is not None:
            self._play_android(path)
        else:
            self._play_kivy(path)

    def _play_android(self, path: str):
        player = MediaPlayer()
        player.setDataSource(path)
        player.setAudioStreamType(AudioManager.STREAM_MUSIC)
        player.prepare()
        with self._lock:
            self._current_player = player
        player.start()
        while player.isPlaying():
            time.sleep(0.05)
        player.release()
        with self._lock:
            self._current_player = None

    def _play_kivy(self, path: str):
        from kivy.core.audio import SoundLoader
        from kivy.clock import Clock

        sound = SoundLoader.load(path)
        if sound is None:
            return

        done = threading.Event()
        sound.bind(on_stop=lambda _: done.set())

        with self._lock:
            self._current_player = sound
        sound.play()

        while not done.is_set():
            time.sleep(0.05)

        with self._lock:
            self._current_player = None

    def cleanup(self):
        self.stop()
        try:
            import shutil
            shutil.rmtree(self._dir, ignore_errors=True)
        except Exception:
            pass
