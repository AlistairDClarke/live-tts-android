#!/usr/bin/env python
"""
LiveTTS Reader — Android/Kivy Edition
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window

import core.kokoro_engine
try:
    import core.chatterbox_engine
    import core.chatterbox_full_engine
except ImportError:
    pass
try:
    import numpy as np
except ImportError:
    pass
from core.tts_engine import TTSFactory
from core.text_processor import chunk_text, normalize_text, find_chunk_at_position
from core.ebook_parser import parse_ebook
from core.voice_manager import VoiceManager
from core.android_audio import AudioService


class HighlightableLabel(Label):
    """A Label that supports click-to-select, highlights, and font sizing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.font_size = 16
        self.halign = "left"
        self.valign = "top"
        self.text_size = (Window.width - 40, None)
        self.bind(size=self._update_text_size)
        self._selected = None
        self._chunks = []
        self._hl_colors = {}
        self._hl_timer = None

    def _update_text_size(self, *args):
        self.text_size = (self.width - 20, None)

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        if self._hl_timer is None:
            self._hl_timer = Clock.schedule_interval(self._render_hl, 0.05)
        # Calculate click position
        line_height = self.font_size * 1.5
        scroll_offset = self.parent.scroll_y if hasattr(self, 'parent') and self.parent else 0
        char_idx = int((touch.pos[1] - self.y) / line_height * 60 + touch.pos[0] / (self.font_size * 0.6))
        char_idx = max(0, min(char_idx, len(self.text) - 1))
        self.parent.parent.parent._on_text_click(char_idx)
        return True

    def set_highlight(self, start, end, color):
        self._hl_colors[color] = (start, end)

    def clear_highlights(self):
        self._hl_colors.clear()

    def _render_hl(self, dt):
        # In Kivy, we can't easily highlight specific text ranges without markup.
        # We'll track highlights as data and render them in the parent via canvas.
        pass


class PlaybackBar(BoxLayout):
    """Play/Pause, Stop, speed, buffer controls."""

    def __init__(s, app_ref, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=0.06, **kwargs)
        s.app = app_ref

        s._play_btn = Button(text="\u25b6", size_hint_x=0.1)
        s._play_btn.bind(on_press=lambda _: s.app._toggle_playback())
        s.add_widget(s._play_btn)

        s._stop_btn = Button(text="\u25a0", size_hint_x=0.08)
        s._stop_btn.bind(on_press=lambda _: s.app._stop())
        s.add_widget(s._stop_btn)

        s.add_widget(Label(text="Spd:", size_hint_x=0.08))
        s._speed = Slider(min=0.5, max=2.0, value=1.0, step=0.1, size_hint_x=0.2)
        s._speed.bind(value=lambda _, v: setattr(s.app, '_speed', v))
        s.add_widget(s._speed)

        s.add_widget(Label(text="Buf:", size_hint_x=0.07))
        s._bufmin = Slider(min=1, max=8, value=2, step=1, size_hint_x=0.15)
        s._bufmin.bind(value=lambda _, v: setattr(s.app, '_bufsize', int(v)))
        s.add_widget(s._bufmin)

        s.add_widget(Label(text="Max:", size_hint_x=0.07))
        s._bufmax = Slider(min=2, max=32, value=16, step=1, size_hint_x=0.15)
        s._bufmax.bind(value=lambda _, v: setattr(s.app, '_bufmax', int(v)))
        s.add_widget(s._bufmax)


class VoicePanel(BoxLayout):
    """Voice selection panel."""

    def __init__(s, app_ref, **kwargs):
        super().__init__(orientation="vertical", size_hint_x=0.25, **kwargs)
        s.app = app_ref
        tabs = TabbedPanel(do_default_tab=False)

        # Selection tab
        sel = TabbedPanelItem(text="Voice")
        sel_layout = BoxLayout(orientation="vertical")
        s._voice_spinner = Spinner(text="(no voices)", size_hint_y=0.08)
        s._voice_spinner.bind(text=lambda _, v: s.app._on_voice_change(v))
        sel_layout.add_widget(s._voice_spinner)
        s._voice_info = Label(text="", size_hint_y=0.3, font_size=12)
        sel_layout.add_widget(s._voice_info)
        sel.content = sel_layout
        tabs.add_widget(sel)

        # Settings tab
        sett = TabbedPanelItem(text="Settings")
        sett_layout = BoxLayout(orientation="vertical")
        sett_layout.add_widget(Label(text="Exaggeration:", size_hint_y=0.1))
        s._exag = Slider(min=0, max=1, value=0.5, step=0.05, size_hint_y=0.1)
        sett_layout.add_widget(s._exag)
        sett_layout.add_widget(Label(text="CFG Weight:", size_hint_y=0.1))
        s._cfg = Slider(min=0, max=1, value=0.5, step=0.05, size_hint_y=0.1)
        sett_layout.add_widget(s._cfg)
        sett_layout.add_widget(Label(size_hint_y=0.7))
        sett.content = sett_layout
        tabs.add_widget(sett)

        s.add_widget(tabs)


class LibraryPanel(BoxLayout):
    """Book library sidebar."""

    def __init__(s, app_ref, **kwargs):
        super().__init__(orientation="vertical", size_hint_x=0.2, **kwargs)
        s.app = app_ref
        s.add_widget(Label(text="Library", size_hint_y=0.06, bold=True))
        s._list = BoxLayout(orientation="vertical", size_hint_y=0.85)
        scroll = ScrollView(size_hint_y=0.85)
        scroll.add_widget(s._list)
        s.add_widget(scroll)
        s._open_btn = Button(text="Open Book...", size_hint_y=0.06)
        s._open_btn.bind(on_press=lambda _: s.app._open_book())
        s.add_widget(s._open_btn)

    def add_book(self, title):
        btn = Button(text=title[:40], size_hint_y=None, height=40, halign="left")
        btn.bind(on_press=lambda _, t=title: self.app._load_book(t))
        self._list.add_widget(btn)

    def clear(self):
        self._list.clear_widgets()


class LiveTTSApp(App):
    """Main Kivy application."""

    def build(self):
        self._engine = None
        self._chunks = []
        self._worker = None
        self._audio = AudioService()
        self._voice_manager = VoiceManager.instance()
        self._speed = 1.0
        self._bufsize = 2
        self._bufmax = 16
        self._cpos = -1
        self._paused = False
        self._playing = False
        self._plist = []
        self._pcur = -1
        self._bufcnt = 0
        self._buf_evt = threading.Event()

        root = BoxLayout(orientation="vertical")

        # Playback bar
        self._playback = PlaybackBar(self)
        root.add_widget(self._playback)

        # Main area: library + reader + voice
        main = BoxLayout(orientation="horizontal")

        self._library = LibraryPanel(self)
        main.add_widget(self._library)

        # Reader (scrollable text)
        reader_scroll = ScrollView(size_hint_x=0.55)
        self._reader = HighlightableLabel(
            text="Open a book or tap to begin.\n\n"
                 "Select a starting position, then press Play.",
            markup=False, size_hint_y=None,
        )
        self._reader.bind(texture_size=self._reader.setter("size"))
        reader_scroll.add_widget(self._reader)
        main.add_widget(reader_scroll)

        self._voice_panel = VoicePanel(self)
        main.add_widget(self._voice_panel)

        root.add_widget(main)

        # Status bar
        self._status = Label(text="Initializing engine...", size_hint_y=0.05)
        root.add_widget(self._status)

        # Init engine
        Clock.schedule_once(lambda dt: self._init_engine(), 0.5)
        return root

    def _init_engine(self):
        self._status.text = "Loading Kokoro..."
        try:
            self._engine = TTSFactory.create("kokoro", device="cpu")
            self._populate_voices()
            self._status.text = "Ready — select text, then Play"
        except Exception as e:
            self._status.text = f"Engine failed: {e}"

    def _populate_voices(self):
        if self._engine is None:
            return
        voices = self._engine.list_voices()
        if getattr(self._engine, '_pipeline', None):
            voices = [v for v in voices if v.name.startswith(("a", "b"))]
        self._voice_panel._voice_spinner.values = [v.name for v in voices]
        if voices:
            self._voice_panel._voice_spinner.text = voices[0].name

    def _on_voice_change(self, name):
        pass  # Voice selection updated via spinner

    def _on_text_click(self, pos):
        self._cpos = pos
        if self._playing:
            self._stop()
            self._status.text = "Stopped — tap Play for new position"
        self._update_selection(pos)

    def _update_selection(self, pos):
        if not self._reader.text.strip():
            return
        tx = self._reader.text
        self._chunks = chunk_text(tx)
        si = find_chunk_at_position(self._chunks, pos)
        if 0 <= si < len(self._chunks):
            ch = self._chunks[si]
            self._status.text = f"Selected: {ch.text[:40]}..."

    def _toggle_playback(self):
        if self._paused:
            self._resume()
        elif self._playing:
            self._pause()
        else:
            self._play()

    def _play(self):
        if self._engine is None:
            self._status.text = "No engine loaded"
            return
        tx = self._reader.text
        if not tx.strip():
            self._status.text = "No text to play"
            return

        pos = self._cpos if self._cpos >= 0 else 0
        self._chunks = chunk_text(tx)
        si = find_chunk_at_position(self._chunks, pos)
        chunks = self._chunks[si:]
        if not chunks:
            self._status.text = "No text beyond cursor"
            return

        self._playing = True
        self._paused = False
        self._plist = []
        self._pcur = -1
        self._bufcnt = 0
        self._worker = threading.Thread(target=self._generator_loop, args=(chunks,), daemon=True)
        self._worker.start()
        self._playback._play_btn.text = "\u23f8"
        self._status.text = "Playing..."

    def _generator_loop(self, chunks):
        import wave, tempfile
        for c in chunks:
            tts_text = normalize_text(c.text)
            output = self._engine.generate(tts_text, speed=self._speed)
            fd, path = tempfile.mkstemp(suffix=".wav")
            d = np.ascontiguousarray(output.audio, dtype=np.float32)
            d16 = np.clip(d * 32767, -32768, 32767).astype(np.int16)
            with os.fdopen(fd, "wb") as f:
                with wave.open(f, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(output.sample_rate)
                    wf.writeframes(d16.tobytes())
            self._audio.play(output.audio, output.sample_rate)
            try:
                os.unlink(path)
            except Exception:
                pass
        Clock.schedule_once(lambda dt: self._on_done(), 0)

    def _on_done(self):
        self._playing = False
        self._paused = False
        self._playback._play_btn.text = "\u25b6"
        self._status.text = "Ready"

    def _pause(self):
        self._paused = True
        self._audio.stop()
        self._playback._play_btn.text = "\u25b6"
        self._status.text = "Paused"

    def _resume(self):
        self._paused = False
        self._playback._play_btn.text = "\u23f8"
        self._status.text = "Playing..."
        # Resume from current position — simplified for now

    def _stop(self):
        self._playing = False
        self._paused = False
        self._audio.stop()
        self._plist = []
        self._pcur = -1
        self._bufcnt = 0
        self._playback._play_btn.text = "\u25b6"
        self._reader.clear_highlights()
        self._status.text = "Stopped"

    def _open_book(self):
        try:
            from android.storage import primary_external_storage_path as sdcard
            download_dir = os.path.join(sdcard(), "Download")
        except ImportError:
            download_dir = os.path.expanduser("~/Download")
        path = os.path.join(download_dir, "book.epub")
        if os.path.exists(path):
            self._load_book(path)
        else:
            self._status.text = "No book found in Downloads"

    def _load_book(self, path):
        self._status.text = "Loading..."
        try:
            book = parse_ebook(path)
            text = self._build_text(book)
            self._reader.text = text
            self._status.text = f"Loaded: {book.title}"
        except Exception as e:
            self._status.text = f"Error: {e}"

    def _build_text(self, book):
        parts = []
        for ch in book.chapters:
            ct = ch.title.strip()
            if ct:
                parts.append(f"=== {ct} ===\n\n")
            co = ch.content.strip()
            if co:
                parts.append(co + "\n\n")
        return "".join(parts)

    def on_stop(self):
        self._audio.cleanup()
        if self._engine:
            try:
                self._engine.unload()
            except Exception:
                pass
