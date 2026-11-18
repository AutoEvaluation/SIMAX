"""Pause and chunk planning (no audio libraries or models needed beyond numpy)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from simax.audio.common import chunk_turns, natural_gaps

TURNS = [{"speaker": "doctor" if i % 2 == 0 else "patient", "text": "word " * (20 + i % 7)} for i in range(120)]


class AudioPlanTest(unittest.TestCase):
    def test_pauses_are_seeded_by_the_sample_id(self):
        self.assertEqual(natural_gaps(TURNS, "a"), natural_gaps(TURNS, "a"))
        self.assertNotEqual(natural_gaps(TURNS, "a"), natural_gaps(TURNS, "b"))

    def test_chunks_cover_every_turn_once_and_respect_the_word_limit(self):
        chunks = chunk_turns(TURNS, natural_gaps(TURNS, "a"))
        self.assertEqual(chunks[0][0], 0)
        self.assertEqual(chunks[-1][1], len(TURNS))
        for (_, end), (start, _) in zip(chunks, chunks[1:]):
            self.assertEqual(end, start)
        for start, end in chunks:
            self.assertLessEqual(sum(len(t["text"].split()) for t in TURNS[start:end]), 700)


if __name__ == "__main__":
    unittest.main()
