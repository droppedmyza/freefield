import numpy as np
import pytest

from freefield import media
from freefield.errors import EffectError


def test_clean_text_collapses_spaces_and_cuts_to_the_limit():
    assert media.clean_text("  Summer\n  sale  ") == "Summer sale"
    assert media.clean_text(None) == "" and media.clean_text("x" * 50) == "x" * 40
    assert media.clean_text("BONK BONK BONK", 12) == "BONK BONK BO"


def test_blend_paints_the_layer_where_it_is_opaque_only():
    frame = np.full((4, 4, 3), 10, np.uint8)
    layer = np.zeros((4, 4, 4), np.float32)
    layer[:2, :, :3], layer[:2, :, 3] = 1.0, 1.0
    out = media.blend(frame, layer, 0.5)
    assert out.dtype == np.uint8
    assert (out[:2] == round(10 * 0.5 + 255 * 0.5)).all() and (out[2:] == 10).all()


def test_fit_cover_fills_the_frame_without_bars():
    wide = np.zeros((100, 400, 3), np.uint8)
    wide[:, 150:250] = 255
    out = media.fit_cover(wide)
    assert out.shape == (media.H, media.W, 3) and out.min() == 255


def test_encoded_video_reads_back_and_a_failed_encode_raises(tmp_path):
    frames = [np.full((64, 48, 3), v, np.uint8) for v in (0, 120, 240)]
    clip = media.encode(frames, tmp_path / "c.mp4", [], ["-c:v", "libx264", "-pix_fmt", "yuv420p"])
    back = media.video_frames_bgr(clip)
    assert len(back) == 3 and abs(int(back[1].mean()) - 120) < 6
    assert len(media.video_frames_bgr(clip, limit=2)) == 2
    with pytest.raises(EffectError, match="ffmpeg failed"):
        media.encode(frames, tmp_path / "c.mp4", [], ["-c:v", "no-such-codec"])
