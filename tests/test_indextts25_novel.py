from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

from indextts25_novel import SegmentSettings, create_or_load_manifest, render_pending_segments, split_text, wave_info


def write_wav(path: Path, frames: int = 2205) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22_050)
        handle.writeframes(b"\x00\x00" * frames)


class FakeIndexTTS:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def infer(self, **kwargs):
        self.calls.append(kwargs)
        write_wav(Path(kwargs["output_path"]))
        return kwargs["output_path"]


class IndexTTSNovelTests(unittest.TestCase):
    def test_split_preserves_text_and_boundaries(self) -> None:
        settings = SegmentSettings(target_chars=24, hard_chars=32)
        text = "第一句很短。第二句也很短。\n\n这是一段明显超过上限的句子，需要在逗号或者安全字符边界处分开，才能避免模型一次收到过长文本。"
        chunks = split_text(text, settings)
        self.assertGreaterEqual(len(chunks), 3)
        self.assertTrue(all(0 < len(chunk.text) <= settings.hard_chars for chunk in chunks))
        self.assertEqual("".join(chunk.text for chunk in chunks).replace(" ", ""), text.replace("\n", "").replace(" ", ""))

    def test_manifest_reopens_only_when_inputs_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            book = root / "book.txt"
            reference = root / "voice.wav"
            book.write_text("第一句。第二句。", encoding="utf-8")
            write_wav(reference)
            settings = SegmentSettings()
            first = create_or_load_manifest(root / "job", book, reference, settings)
            second = create_or_load_manifest(root / "job", book, reference, settings)
            self.assertEqual(first["config_signature"], second["config_signature"])
            book.write_text("换了内容。", encoding="utf-8")
            with self.assertRaises(ValueError):
                create_or_load_manifest(root / "job", book, reference, settings)

    def test_renderer_writes_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            book = root / "book.txt"
            reference = root / "voice.wav"
            book.write_text("这是第一句，需要单独生成。这是第二句，也需要单独生成。这是第三句，最后完成生成。", encoding="utf-8")
            write_wav(reference)
            settings = SegmentSettings(target_chars=20, hard_chars=24)
            job = root / "job"
            manifest = create_or_load_manifest(job, book, reference, settings)
            fake = FakeIndexTTS()
            partial = render_pending_segments(job, manifest, fake, reference, max_segments=1)
            self.assertEqual(partial["status"], "partial")
            self.assertEqual(len(fake.calls), 1)
            completed = render_pending_segments(job, partial, fake, reference)
            self.assertEqual(completed["status"], "rendered")
            self.assertEqual(len(fake.calls), len(completed["segments"]))
            for record in completed["segments"]:
                self.assertIsNotNone(wave_info(job / record["output_relpath"]))


if __name__ == "__main__":
    unittest.main()
