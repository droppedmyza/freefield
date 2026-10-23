from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS = Path(__file__).resolve().parent / "fonts"
FONT = FONTS / "LuckiestGuy-Regular.ttf"
SERIF = FONTS / "PlayfairDisplay-Variable.ttf"
STYLES = {
    "toy": {"card": "#d7141a", "fill": "#ffd400", "edge": "#ffd400"},
    "magazine": {"card": "#0d0d0d", "fill": "#ffffff", "edge": "#e5002b"},
}
SIZE = (720, 192)


def fit_font(text: str, max_w: int, max_h: int) -> ImageFont.FreeTypeFont:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    size = max_h
    while size > 8:
        font = ImageFont.truetype(str(FONT), size)
        box = draw.textbbox((0, 0), text, font=font, stroke_width=max(2, size // 9))
        if box[2] - box[0] <= max_w and box[3] - box[1] <= max_h:
            return font
        size -= 2
    return ImageFont.truetype(str(FONT), 8)


def lettering(text: str, font: ImageFont.FreeTypeFont, fill: str, outline: str, shadow: str, tilt: float) -> Image.Image:
    size = font.size
    stroke, drop = max(2, size // 9), max(2, size // 12)
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    l, t, r, b = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    img = Image.new("RGBA", (r - l + drop + 4, b - t + drop + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((2 - l + drop, 2 - t + drop), text, font=font, fill=shadow, stroke_width=stroke, stroke_fill=shadow)
    d.text((2 - l, 2 - t), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=outline)
    return img.rotate(tilt, resample=Image.BICUBIC, expand=True) if tilt else img


def render(title: str, subtitle: str, card: str, fill: str, edge: str) -> Image.Image:
    w, h = SIZE
    img = Image.new("RGB", SIZE, card)
    d = ImageDraw.Draw(img)
    for y in range(h):
        k = 1.0 + 0.18 * (0.5 - y / h)
        r, g, b = Image.new("RGB", (1, 1), card).getpixel((0, 0))
        d.line([(0, y), (w, y)], fill=(min(255, int(r * k)), min(255, int(g * k)), min(255, int(b * k))))
    d.rectangle([0, h - 10, w, h], fill=edge)
    d.rounded_rectangle([w // 2 - 46, 10, w // 2 + 46, 28], radius=9, fill="#2a0606")
    sub_h = 26 if subtitle else 0
    top, bottom = 34, h - 14 - sub_h
    word = lettering(title, fit_font(title, int(w * 0.86), bottom - top), fill, "#111111", "#5a0000", 3.0)
    img.paste(word, ((w - word.width) // 2, top + (bottom - top - word.height) // 2), word)
    if subtitle:
        font = ImageFont.truetype(str(FONT), 22)
        tw = d.textlength(subtitle, font=font)
        d.text(((w - tw) / 2, h - 14 - sub_h + 2), subtitle, font=font, fill="#ffffff", stroke_width=2, stroke_fill="#111111")
    return img


def serif(size: int, weight: str) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(SERIF), size)
    font.set_variation_by_name(weight)
    return font


def render_magazine(title: str, subtitle: str, card: str, fill: str, edge: str) -> Image.Image:
    w, h = SIZE
    img = Image.new("RGB", SIZE, card)
    d = ImageDraw.Draw(img)
    sub_h = 34 if subtitle else 0
    max_h, size = h - 18 - sub_h - 8, h
    while size > 8:
        font = serif(size, "Black")
        l, t, r, b = d.textbbox((0, 0), title, font=font)
        if r - l <= int(w * 0.9) and b - t <= max_h:
            break
        size -= 2
    d.text(((w - (r - l)) / 2 - l, 8 + (max_h - (b - t)) / 2 - t), title, font=font, fill=fill)
    if subtitle:
        font = serif(24, "Bold")
        d.text(((w - d.textlength(subtitle, font=font)) / 2, h - 6 - sub_h + 2), subtitle, font=font, fill=edge)
    d.rectangle([0, h - 6, w, h], fill=edge)
    return img


def write_header(title: str, out: Path, style: str = "toy", subtitle: str = "", **colors: str | None) -> Path:
    if not title.strip():
        raise ValueError("title is empty")
    chosen = {k: colors.get(k) or v for k, v in STYLES[style].items()}
    draw = render_magazine if style == "magazine" else render
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(title.strip(), subtitle.strip(), **chosen).save(out)
    return out
