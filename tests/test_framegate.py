"""Tests for the keyframe gate.

The failure that matters is not "does it dedupe" — it is "does it drop something
that mattered". Every test that asserts a reduction is paired with one asserting
a real change survives it.
"""

from __future__ import annotations

import pytest
from PIL import Image

from audrey.media.framegate import (
    DEFAULT_MAX_RUN,
    Keyframe,
    dhash,
    hamming,
    main,
    select,
    select_keyframes,
)


def textured(seed: int = 0, size: int = 64) -> Image.Image:
    """A deterministic blocky pattern.

    Test images need real structure or they are indistinguishable to the gate:
    dHash asks "is this pixel brighter than the one to its right", so a flat field
    and a left-to-right ramp both answer "no" everywhere and hash to all zeros.
    Blocks survive the downscale to 9x8 that hashing does.
    """
    cells, step = 8, size // 8
    values = [
        [(seed * 7919 + x * 104729 + y * 1299709) * 2654435761 % 251
         for x in range(cells)]
        for y in range(cells)
    ]
    data = bytes(values[y // step][x // step] for y in range(size) for x in range(size))
    return Image.frombytes("L", (size, size), data).convert("RGB")


def noisy(base: Image.Image, amount: int = 3) -> Image.Image:
    """Same picture, different bytes — sensor noise between two frames."""
    grey = base.convert("L")
    data = bytes(min(255, p + (i % amount)) for i, p in enumerate(grey.tobytes()))
    return Image.frombytes("L", grey.size, data).convert("RGB")


class TestHash:
    def test_identical_images_hash_identically(self):
        assert dhash(textured(1)) == dhash(textured(1))

    def test_hash_is_64_bits(self):
        assert dhash(textured(1)).bit_length() <= 64

    def test_sensor_noise_does_not_move_the_hash_much(self):
        """The case sha256 caching in vision.py cannot catch."""
        base = textured(1)
        assert hamming(dhash(base), dhash(noisy(base))) < 8

    def test_a_different_picture_moves_the_hash(self):
        assert hamming(dhash(textured(1)), dhash(textured(2))) >= 8

    def test_hamming_is_symmetric_and_zero_on_self(self):
        assert hamming(0b1011, 0b1011) == 0
        assert hamming(0b1011, 0b0010) == hamming(0b0010, 0b1011) == 2


class TestSelect:
    def test_no_frames_selects_nothing(self):
        assert select([]) == []

    def test_the_first_frame_is_always_kept(self):
        """It has nothing prior to be redundant with."""
        chosen = select([0b1111, 0b1111, 0b1111], min_distance=1)
        assert chosen[0].index == 0

    def test_identical_frames_collapse_to_one(self):
        chosen = select([0xFF00] * 19, min_distance=8)
        assert len(chosen) == 1
        assert chosen[0].span == 19

    def test_distinct_frames_are_all_kept(self):
        chosen = select([0x0000, 0xFFFF, 0x0000, 0xFFFF], min_distance=8)
        assert [f.index for f in chosen] == [0, 1, 2, 3]

    def test_represents_covers_every_frame_exactly_once(self):
        """A frame that is neither kept nor attributed has silently vanished."""
        hashes = [0x0, 0x0, 0x0, 0xFFFF, 0xFFFF, 0x0]
        chosen = select(hashes, min_distance=8)
        covered = [i for frame in chosen for i in frame.represents]
        assert sorted(covered) == list(range(len(hashes)))
        assert len(covered) == len(set(covered))

    def test_a_kept_frame_represents_itself(self):
        chosen = select([0x0, 0x0], min_distance=8)
        assert chosen[0].index in chosen[0].represents

    def test_slow_drift_is_eventually_kept(self):
        """The reason comparison is against the last kept frame, not the previous one.

        Each step differs from its neighbour by one bit — under pairwise
        comparison the whole sequence collapses and the endpoint, which looks
        nothing like the start, never gets described.
        """
        hashes = [(1 << n) - 1 for n in range(0, 20)]  # 0b0, 0b1, 0b11, 0b111, ...
        chosen = select(hashes, min_distance=4, max_run=1000)
        assert len(chosen) > 1
        assert max(hamming(hashes[0], hashes[f.index]) for f in chosen) >= 4

    def test_max_run_forces_coverage_of_a_static_stretch(self):
        chosen = select([0x1234] * 100, min_distance=8, max_run=20)
        assert len(chosen) == 5
        assert all(f.span <= 20 for f in chosen)

    def test_max_run_does_not_split_what_already_changed(self):
        chosen = select([0x0, 0xFFFF], min_distance=8, max_run=DEFAULT_MAX_RUN)
        assert [f.span for f in chosen] == [1, 1]

    def test_min_distance_zero_keeps_everything(self):
        """Any two frames differ by at least 0 bits, so nothing is ever dropped."""
        assert len(select([0x1] * 5, min_distance=0)) == 5

    @pytest.mark.parametrize("bad", [-1])
    def test_negative_min_distance_is_refused(self, bad):
        with pytest.raises(ValueError, match="min_distance"):
            select([0x1], min_distance=bad)

    @pytest.mark.parametrize("bad", [0, -3])
    def test_max_run_below_one_is_refused(self, bad):
        """A zero run would attribute frames to a keyframe that covers none."""
        with pytest.raises(ValueError, match="max_run"):
            select([0x1], max_run=bad)


class TestSelectKeyframes:
    def write(self, tmp_path, images):
        paths = []
        for n, img in enumerate(images):
            p = tmp_path / f"f{n:03d}.png"
            img.save(p)
            paths.append(p)
        return paths

    def test_a_static_shot_costs_one_describe_call(self, tmp_path):
        base = textured(1)
        paths = self.write(tmp_path, [base] + [noisy(base) for _ in range(18)])
        chosen = select_keyframes(paths)
        assert len(chosen) == 1
        assert chosen[0].span == 19

    def test_a_cut_survives_the_gate(self, tmp_path):
        """The pairing for the test above: dedupe must not eat a scene change."""
        paths = self.write(tmp_path, [textured(1), textured(1), textured(2)])
        chosen = select_keyframes(paths)
        assert len(chosen) == 2
        assert chosen[1].index == 2

    def test_paths_are_carried_through(self, tmp_path):
        paths = self.write(tmp_path, [textured(1), textured(1)])
        assert select_keyframes(paths)[0].path == paths[0]

    def test_no_frames_is_not_an_error(self, tmp_path):
        assert select_keyframes([]) == []


class TestCli:
    def test_it_reports_the_reduction(self, tmp_path, capsys):
        base = textured(1)
        paths = [tmp_path / f"f{n}.png" for n in range(4)]
        base.save(paths[0])
        for p in paths[1:]:
            noisy(base).save(p)

        assert main([str(p) for p in paths]) == 0
        out = capsys.readouterr().out
        assert "1/4 frames need a describe call" in out
        assert "75% fewer GPU calls" in out


class TestKeyframe:
    def test_span_counts_the_frames_covered(self):
        assert Keyframe(index=0, represents=(0, 1, 2)).span == 3
