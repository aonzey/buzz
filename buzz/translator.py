import os
import re
import logging
import queue
import threading

from typing import Callable, Optional, List, Tuple
from openai import OpenAI, max_retries
from PyQt6.QtCore import QObject, pyqtSignal

from buzz.locale import _
from buzz.settings.settings import Settings
from buzz.store.keyring_store import get_password, Key
from buzz.transcriber.transcriber import TranscriptionOptions
from buzz.widgets.transcriber.advanced_settings_dialog import AdvancedSettingsDialog


BATCH_SIZE = 10
# Local LLMs can be slow, give them plenty of time to respond
TRANSLATION_TIMEOUT_SECONDS = 600.0


def build_openai_client() -> OpenAI:
    """Build the OpenAI-compatible client used for translations."""
    from buzz.proxy import apply_proxy

    # The OpenAI client reads HTTP(S)_PROXY from the environment, so exporting
    # the configured proxy here is enough to route translation requests.
    apply_proxy()

    settings = Settings()
    custom_openai_base_url = os.getenv(
        "BUZZ_TRANSLATION_API_BASE_URl",
        settings.value(key=Settings.Key.CUSTOM_OPENAI_BASE_URL, default_value=""),
    )
    openai_api_key = os.getenv("BUZZ_TRANSLATION_API_KEY", get_password(Key.OPENAI_API_KEY))
    return OpenAI(
        api_key=openai_api_key,
        base_url=custom_openai_base_url if custom_openai_base_url else None,
        max_retries=0,
    )


class Translator(QObject):
    translation = pyqtSignal(str, int)
    finished = pyqtSignal()

    def __init__(
        self,
        transcription_options: TranscriptionOptions,
        advanced_settings_dialog: AdvancedSettingsDialog,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)

        logging.debug(f"Translator init: {transcription_options}")

        self.transcription_options = transcription_options
        self.advanced_settings_dialog = advanced_settings_dialog
        self.advanced_settings_dialog.transcription_options_changed.connect(
            self.on_transcription_options_changed
        )

        self.queue = queue.Queue()

        # Guards the pause flag, the worker waits on it between requests
        self._pause_condition = threading.Condition()
        self._paused = False

        self.openai_client = build_openai_client()

    def _translate_single(self, transcript: str, transcript_id: int) -> Tuple[str, int]:
        """Translate a single transcript via the API. Returns (translation, transcript_id)."""
        translation = translate_single(
            self.openai_client,
            self.transcription_options.llm_model,
            self.transcription_options.llm_prompt,
            transcript,
        )
        return translation, transcript_id

    def _translate_batch(self, items: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
        """Translate multiple transcripts in a single API call.
        Returns list of (translation, transcript_id) in the same order as input."""
        translations = translate_batch(
            self.openai_client,
            self.transcription_options.llm_model,
            self.transcription_options.llm_prompt,
            [transcript for transcript, _ in items],
        )
        return [
            (translations[i] if i < len(translations) else "", transcript_id)
            for i, (_, transcript_id) in enumerate(items)
        ]

    @staticmethod
    def _parse_batch_response(response: str, expected_count: int) -> List[str]:
        """Parse a numbered batch response like '[1] text\\n[2] text' into a list of strings."""
        # Split on [N] markers — re.split with a group returns: [before, group1, after1, group2, after2, ...]
        parts = re.split(r'\[(\d+)\]\s*', response)

        translations = {}
        for i in range(1, len(parts) - 1, 2):
            num = int(parts[i])
            text = parts[i + 1].strip()
            translations[num] = text

        return [
            translations.get(i, "")
            for i in range(1, expected_count + 1)
        ]

    def _wait_while_paused(self):
        """Block the worker until the translation is resumed."""
        with self._pause_condition:
            while self._paused:
                self._pause_condition.wait()

    def pause(self):
        """Stop picking up new segments until resume() is called."""
        with self._pause_condition:
            self._paused = True

    def resume(self):
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()

    def is_paused(self) -> bool:
        with self._pause_condition:
            return self._paused

    def clear_queue(self):
        """Drop the segments that have not been sent to the API yet."""
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                return

    def cancel(self):
        """Discard the pending work without stopping the worker thread.

        A segment already in flight still comes back, the caller is expected
        to ignore results once it lost interest in them.
        """
        logging.debug("Cancelling queued translations")
        self.resume()
        self.clear_queue()

    def start(self):
        logging.debug("Starting translation queue")

        while True:
            self._wait_while_paused()
            item = self.queue.get()  # Block until item available

            # Check for sentinel value (None means stop)
            if item is None:
                logging.debug("Translation queue received stop signal")
                break

            # Collect a batch: start with the first item, then drain more
            batch = [item]
            stop_after_batch = False
            while len(batch) < BATCH_SIZE:
                try:
                    next_item = self.queue.get_nowait()
                    if next_item is None:
                        stop_after_batch = True
                        break
                    batch.append(next_item)
                except queue.Empty:
                    break

            if len(batch) == 1:
                transcript, transcript_id = batch[0]
                translation, tid = self._translate_single(transcript, transcript_id)
                self.translation.emit(translation, tid)
            else:
                logging.debug(f"Translating batch of {len(batch)} in single request")
                results = self._translate_batch(batch)
                for translation, tid in results:
                    self.translation.emit(translation, tid)

            if stop_after_batch:
                logging.debug("Translation queue received stop signal")
                break

        logging.debug("Translation queue stopped")
        self.finished.emit()

    def on_transcription_options_changed(
        self, transcription_options: TranscriptionOptions
    ):
        self.transcription_options = transcription_options

    def enqueue(self, transcript: str, transcript_id: Optional[int] = None):
        self.queue.put((transcript, transcript_id))

    def stop(self):
        # Never leave the worker blocked on the pause condition, it would
        # never read the sentinel value below
        self.resume()
        # Send sentinel value to unblock and stop the worker thread
        self.queue.put(None)


def _batch_prompt(prompt: str, count: int) -> str:
    return (
        f"{prompt}\n\n"
        f"You will receive {count} numbered texts. "
        f"Process each one separately according to the instruction above "
        f"and return them in the exact same numbered format, e.g.:\n"
        f"[1] processed text\n[2] processed text"
    )


def translate_single(client: OpenAI, model: str, prompt: str, text: str) -> str:
    """Translate one piece of text. Returns "" when the request fails."""
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            timeout=TRANSLATION_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logging.error(f"Translation error! Server response: {e}")
        return ""

    if completion and completion.choices and completion.choices[0].message:
        return completion.choices[0].message.content or ""
    logging.error(f"Translation error! Server response: {completion}")
    return ""


def translate_batch(
    client: OpenAI, model: str, prompt: str, texts: List[str]
) -> List[str]:
    """Translate several texts in one request, preserving the input order."""
    combined = "\n".join(f"[{i}] {text}" for i, text in enumerate(texts, 1))
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _batch_prompt(prompt, len(texts))},
                {"role": "user", "content": combined},
            ],
            timeout=TRANSLATION_TIMEOUT_SECONDS,
        )
    except Exception as e:
        logging.error(f"Batch translation error! Server response: {e}")
        return [""] * len(texts)

    if not (completion and completion.choices and completion.choices[0].message):
        logging.error(f"Batch translation error! Server response: {completion}")
        return [""] * len(texts)

    response_text = completion.choices[0].message.content
    logging.debug(f"Received batch translation response: {response_text}")
    return Translator._parse_batch_response(response_text, len(texts))


def translate_texts(
    texts: List[str],
    model: str,
    prompt: str,
    batch_size: int = BATCH_SIZE,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[str]:
    """Translate ``texts`` without any Qt/GUI involvement.

    Used by the command line "transcribe&translate" task, which runs on the
    transcription worker thread. Returns one entry per input text ("" when a
    request fails), in the original order. ``on_progress`` is called with
    ``(done, total)`` after every request.
    """
    client = build_openai_client()
    results: List[str] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        chunk = texts[start : start + batch_size]
        if len(chunk) == 1:
            results.append(translate_single(client, model, prompt, chunk[0]))
        else:
            results.extend(translate_batch(client, model, prompt, chunk))

        if on_progress is not None:
            on_progress(len(results), total)

    return results
