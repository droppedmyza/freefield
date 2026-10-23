import numpy as np
import pytest

from freefield import clip_breakout as rb

HOLE = (210, 347, 300, 455)


def person(h=960, w=540, top=300, x0=170, x1=370):
    a = np.zeros((h, w), np.float32)
    a[top:, x0:x1] = 1.0
    return a


def test_frame_0_sits_in_the_opening_like_a_portrait():
    a0 = person()
    f = rb.fit(a0, (210, 330, 120, 160), HOLE)
    x, y, w, h = HOLE
    head_top = f.oy + f.s * 300
    assert head_top == pytest.approx(y + 0.10 * h, abs=0.5)
    assert f.s * 160 == pytest.approx(0.34 * h, abs=0.5)
    assert f.ox + f.s * 270 == pytest.approx(x + w / 2, abs=0.5)
    assert f.oy + f.s * 960 >= y + h


@pytest.mark.parametrize("a0, face, reason", [
    (np.zeros((960, 540), np.float32), (200, 300, 100, 140), "no person"),
    (person(top=600), (200, 640, 140, 300), "further away"),
])
def test_frames_that_cannot_be_placed_say_why(a0, face, reason):
    with pytest.raises(rb.FitError, match=reason):
        rb.fit(a0, face, HOLE)


def test_a_person_who_leans_in_comes_out_over_the_frame_and_is_cut_below_it():
    plate = np.full((1280, 720, 3), 40, np.uint8)
    f = rb.fit(person(), (210, 330, 120, 160), HOLE)
    big = np.zeros((960, 540), np.float32)
    big[100:, 40:500] = 1.0
    frame = np.full((960, 540, 3), 200, np.uint8)
    fade = rb.edge_fade((960, 540), f, (720, 1280))
    out = rb.composite(plate, frame, big, f, fade, HOLE, shadow=0)
    x, y, w, h = HOLE
    assert out[y + h // 2, x - 20].mean() > 150
    assert out[y + h + 5:, :].max() == 40


def test_the_reveal_lifts_the_paint_off_progressively():
    m = rb.reveal_masks((128, 72), 12, seed=3)
    means = m.reshape(12, -1).mean(1)
    assert means[0] < 0.1 and means[-1] > 0.95 and np.all(np.diff(means) >= 0)


def test_a_one_frame_matte_glitch_is_smoothed_away():
    ms = [np.zeros((40, 30), np.float32) for _ in range(5)]
    for m in ms:
        m[10:30, 10:20] = 1
    ms[2][0:10, :] = 1
    out = rb.clean_mattes(ms)
    assert out[2][0:8].max() < 0.1 and out[2][20, 15] > 0.9


def test_a_clip_filmed_too_close_is_told_to_step_back():
    tight = rb.Fit(0.6, 195.0, 271.0)
    wide = rb.Fit(1.4, -20.0, 100.0)
    assert not rb.room_to_break_out(tight, (540, 960), HOLE)
    assert rb.room_to_break_out(wide, (540, 960), HOLE)


def test_a_run_of_frames_that_grab_the_wall_takes_its_neighbours_matte():
    ms = [np.zeros((60, 40), np.float32) for _ in range(15)]
    for m in ms:
        m[20:60, 12:28] = 1
    for i in (5, 6, 7):
        ms[i][0:20, :] = 1
        ms[i][:, 0:12] = 1
    out = rb.clean_mattes(ms)
    assert all(out[i][0:15, 0:8].max() < 0.1 for i in (5, 6, 7))
    assert out[6][40, 20] > 0.9


def test_a_detached_patch_of_the_room_is_dropped():
    m = np.zeros((60, 40), np.float32)
    m[20:60, 12:28] = 1
    m[2:8, 2:8] = 1
    out = rb.clean_mattes([m, m.copy(), m.copy()])
    assert out[1][4, 4] < 0.1 and out[1][40, 20] > 0.9


def test_batch_files_come_back_in_frame_order():
    from pathlib import Path
    names = [f"matte_save_{i}_matte_{i + 1:05d}_.png" for i in range(12)]
    shuffled = sorted(Path(n) for n in names)
    assert [p.name for p in rb.in_order(shuffled)] == names


def plate_composite(alpha):
    plate = np.full((1280, 720, 3), 40, np.uint8)
    frame = np.full((1280, 720, 3), 200, np.uint8)
    return rb.composite(plate, frame, alpha, rb.Fit(1.0, 0.0, 0.0), np.ones((1280, 720), np.float32), HOLE, shadow=0)


def test_an_arm_over_the_gilt_stays_solid():
    x, y, w, h = HOLE
    a = np.zeros((1280, 720), np.float32)
    a[500:, x:x + w] = 1
    a[y + h - 60:y + h - 40, x - 60:x + w] = 1
    out = plate_composite(a)
    assert out[y + h - 50, x - 30].min() == 200
    assert out[y + h - 1, x - 30].max() == 40


def test_a_torso_cut_beside_the_opening_gets_a_soft_edge_not_a_straight_line():
    x, y, w, h = HOLE
    a = np.zeros((1280, 720), np.float32)
    a[600:, x - 60:x + w + 60] = 1
    out = plate_composite(a)
    assert out[y + h - 1, x - 30].mean() < 60
    assert out[y + h - rb.CUT_FADE - 5, x - 30].min() == 200
    assert out[y + h - 1, x + w // 2].min() == 200

def test_the_reveal_sweeps_the_canvas_top_down_and_shows_the_spill_last():
    box = (100, 200, 300, 400)
    m = rb.reveal_masks((800, 500), 10, seed=1, opening=box)
    assert m[4, 220, 250] > 0.9 and m[4, 580, 250] < 0.1
    assert m[5, 100, 20] == 0 and m[-1, 100, 20] == 1

def test_the_room_showing_through_a_gap_is_removed_and_the_person_kept():
    wall, shirt = (230, 230, 225), (160, 30, 40)
    frames, mattes = [], []
    for i in range(12):
        f = np.zeros((96, 64, 3), np.uint8)
        f[:] = wall
        m = np.zeros((96, 64), np.float32)
        x0 = 20 - i
        f[40:, x0:x0 + 20 + 2 * i] = shirt
        m[40:, x0:x0 + 20 + 2 * i] = 1
        if i >= 8:
            m[30:40, 10:54] = 1
        frames.append(f)
        mattes.append(m)
    out, removed = rb.remove_room(frames, mattes)
    assert out[10][34, 30] < 0.1
    assert out[10][60, 25] > 0.9
    assert removed > 0.1


def test_the_room_is_still_recognised_when_the_phone_brightens_it():
    frames, mattes = [], []
    for i in range(12):
        wall = np.array([180 + 4 * i, 180 + 4 * i, 176 + 4 * i], np.uint8)
        f = np.zeros((96, 64, 3), np.uint8)
        f[:] = wall
        m = np.zeros((96, 64), np.float32)
        f[40:, 22:42] = (40, 60, 160)
        m[40:, 22:42] = 1
        if i >= 9:
            m[40:, 14:22] = 1
        frames.append(f)
        mattes.append(m)
    out, _ = rb.remove_room(frames, mattes)
    assert out[10][60, 17] < 0.1 and out[10][60, 30] > 0.9
