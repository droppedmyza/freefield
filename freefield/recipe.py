from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REVEAL_WORDS = ("before/after", "before and after", "before-and-after", "reveal", "wipe")
SPLIT_WORDS = ("side by side", "side-by-side", "split screen", "split-screen", "next to my photo", "next to the original")
LEADS = (r"make me (?:into |look like |a |an )?", r"turn me into ", r"my (?:portrait|photo|selfie) (?:becomes|turns into) ",
         r"i become ", r"you become ", r"me as ", r"make my (?:portrait|photo|selfie) (?:into |a |an )?")
DEFAULT_MOTION = "they slowly turn their head and smile"
KEEP_WORDS = ("stays the same", "stay the same", "stays as it is", "stay as they are", "as it is", "as they are", "unchanged",
              "keep my photo", "keep the photo")
CAPTION_WORDS = ("with a caption", "with my city name", "with a city name", "with the place name", "with a place name",
                 "with a name tag", "with a label")
PET = re.compile(r"\b(dog|cat|puppy|kitten|pet|bunny|rabbit|hamster|parrot|pup)s?\b", re.I)
OBJECT = re.compile(r"\b(toy|product|sneaker|shoe|mug|cup|bottle|plush|plushie|teddy|mascot|figurine|bag|watch|object|lamp)s?\b", re.I)
SCENE = re.compile(r"\b(place|landscape|building|house|city|street|beach|view|skyline|mountain)s?\b", re.I)
HUMAN = re.compile(r"\b(human|person|man|woman|boy|girl|people|character)\b", re.I)
SUBJECT_LEAD = r"(?:my|our|the|this|a)\s+(?:[a-z-]+\s+){0,3}?(?:becomes?|turns? into|is turned into)\s+"


class BuildError(ValueError):
    pass


@dataclass
class Asset:
    name: str
    license: str
    source: str
    path: Path | None = None


@dataclass
class Recipe:
    name: str
    becomes: str
    motion: str = DEFAULT_MOTION
    reveal: bool = False
    split: bool = False
    title: str | None = None
    reference: Asset | None = None
    description: str = ""
    subject: str = "person"
    keep: bool = False
    caption: bool = False
    profile: str = "8gb"

    @property
    def id(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")[:40] or "my-effect"
        return slug + "-studio" if self.profile == "studio" else slug

    def stages(self) -> list[str]:
        return (["transform"] * (not self.keep) + ["animate"] + ["reveal"] * self.reveal + ["split"] * self.split
                + ["title"] * bool(self.title) + ["caption"] * self.caption + ["save"])

    def keeps_subject(self) -> bool:
        return not self.keep and not (self.subject != "person" and HUMAN.search(self.becomes))

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"], d["stages"] = self.id, self.stages()
        if self.reference:
            d["reference"] = {"name": self.reference.name, "license": self.reference.license, "source": self.reference.source}
        return d


def _cut(text: str, words: tuple[str, ...]) -> tuple[str, bool]:
    low, hit = text.lower(), False
    for w in words:
        for m in reversed(list(re.finditer(rf"(?:,?\s*(?:with|and|as)\s+(?:an?\s+)?)?{re.escape(w)}[^,.;]*", low))):
            text, low, hit = text[:m.start()] + text[m.end():], low[:m.start()] + low[m.end():], True
    return text, hit


def third_person(text: str) -> str:
    for pat, rep in ((r"\bI'm\b", "they're"), (r"\bI am\b", "they are"), (r"\bI\b", "they"), (r"\bmyself\b", "themselves"),
                     (r"\bmy\b", "their"), (r"\bme\b", "them"), (r"\bMy\b", "Their")):
        text = re.sub(pat, rep, text)
    return text


def parse(description: str) -> Recipe:
    text = " ".join((description or "").split())
    if not text:
        raise BuildError("describe the effect in a few words, e.g. 'make me a paper cutout, then I pop off the page'")
    if len(text) > 400:
        raise BuildError("keep the description under 400 characters")
    title = None
    m = re.search(r",?\s*(?:titled|called|with the title|caption(?:ed)?)\s*[\"“']([^\"”']{1,40})[\"”']", text, re.I)
    if m:
        title, text = m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()
    named = None
    m = re.search(r",?\s*(?:named|call it|called it)\s*[\"“']([^\"”']{1,40})[\"”']", text, re.I)
    if m:
        named, text = m.group(1).strip(), (text[:m.start()] + text[m.end():]).strip()
    text, caption = _cut(text, CAPTION_WORDS)
    text, reveal = _cut(text, REVEAL_WORDS)
    text, split = _cut(text, SPLIT_WORDS)
    text, keep = _cut(text, KEEP_WORDS)
    head = re.split(r",?\s+then\s+|,?\s+and then\s+", text, maxsplit=1, flags=re.I)[0]
    subject = "pet" if PET.search(head) else "object" if OBJECT.search(head) else "scene" if SCENE.search(head) else "person"
    parts = re.split(r",?\s+then\s+|,?\s+and then\s+", text, maxsplit=1, flags=re.I)
    becomes = parts[0].strip(" ,.")
    for lead in LEADS + (SUBJECT_LEAD,):
        becomes = re.sub(rf"^{lead}", "", becomes, flags=re.I)
    becomes = becomes.strip(" ,.") or text.strip(" ,.")
    motion = third_person(parts[1].strip(" ,.")) if len(parts) > 1 and parts[1].strip(" ,.") else DEFAULT_MOTION
    if keep:
        motion = third_person(parts[1].strip(" ,.")) if len(parts) > 1 and parts[1].strip(" ,.") else third_person(becomes)
        becomes = ""
    words = re.sub(r"^(a|an|the)\s+", "", becomes or motion, flags=re.I).split()[:4]
    while words and words[-1].lower() in {"a", "an", "the", "in", "of", "on", "with", "at", "to", "and", "from", "for", "into", "through"}:
        words.pop()
    name = named or (title.title() if title else " ".join(w.capitalize() for w in words))
    return Recipe(name=name or "My Effect", becomes=becomes, motion=motion, reveal=reveal, split=split, title=title,
                  description=description.strip(), subject=subject, keep=keep, caption=caption)


def keyframe_prompt(r: Recipe) -> str:
    style = " in the style of image 2" if r.reference else ""
    if r.subject == "person":
        return (f"Image 1 is a photo of a person. Make a picture of the exact person from image 1 as {r.becomes}{style}. "
                "Keep the same face, skin tone, age, hair and identity as image 1. One person, vertical portrait framing, "
                "the face clearly visible, high detail, no text, no watermark.")
    kind = {"pet": "pet", "object": "object", "scene": "place"}[r.subject]
    keep = {"pet": "its colors, markings and expression", "object": "its shape, colors and details",
            "scene": "its layout, landmarks and light"}[r.subject]
    if not r.keeps_subject():
        return (f"Image 1 is a blurred colour guide made from a photo of a {kind}. Make a portrait photo of {r.becomes}{style}: "
                "an invented adult whose hair colour echoes the main colours of image 1 and whose clothes echo its other "
                "colours, in a similar setting and light. A fully human face with human ears, a human nose and smooth bare "
                "human skin. A natural friendly expression. An ordinary invented person with an original face. "
                "A head-and-shoulders portrait of one person alone. Photorealistic, vertical portrait "
                "framing, high detail, a clean unmarked photo.")
    if r.subject == "pet":
        return (f"Image 1 is a photo of a pet. Make a new photo of {r.becomes}{style}, made to look like the pet in image 1: "
                f"the same {keep}, now fully changed into its new form and the only subject in the picture. Vertical "
                "framing, high detail, a clean unmarked photo.")
    return (f"Image 1 is a photo of a {kind}. Remake the {kind} from image 1 completely as {r.becomes}{style}: change its "
            f"material and style all the way, but keep {keep} so it is clearly the same {kind}. The {kind} is the only "
            "subject in the picture. Vertical framing, high detail, a clean unmarked photo.")


def motion_prompt(r: Recipe) -> str:
    m = r.motion.rstrip(".")
    still = " Static shot: the camera holds steady and the background stays still."
    if r.keep:
        return f"{m[:1].upper()}{m[1:]}. Smooth, natural motion.{still}"
    head = f"{m[:1].upper()}{m[1:]}. Smooth, natural motion; the subject stays recognisable."
    if r.subject == "person":
        return f"{head} Only the subject moves, alone in the frame; the background stays plain and still. Static shot."
    alone = (" This one person is alone in the picture, seen from the shoulders up. Only the face moves; the shoulders "
             "stay still." if not r.keeps_subject() else "")
    return f"{head}{alone}{still}"


YOU = {"they": "you", "them": "you", "their": "your", "theirs": "yours", "themselves": "yourself", "they're": "you're"}


def describe(r: Recipe) -> str:
    if r.keep:
        text = f"Your {r.subject if r.subject != 'person' else 'photo'}, then {r.motion}."
    elif r.subject == "person":
        becomes = r.becomes if re.match(r"(?:a|an|the|your)\s", r.becomes, re.I) else (
            ("an " if r.becomes[:1].lower() in "aeiou" else "a ") + r.becomes)
        motion = re.sub(r"\b(?:they're|themselves|theirs|their|they|them)\b", lambda m: YOU[m.group(0).lower()], r.motion,
                        flags=re.I)
        text = f"You as {becomes}; then {motion}."
    else:
        text = f"Your {r.subject} as {r.becomes}; then {r.motion}."
    return text + (" With a before/after reveal." if r.reveal else "") + (" Side by side with your photo." if r.split else "")


def recipe_from(data: dict) -> Recipe:
    return Recipe(**{k: v for k, v in data.items() if k in Recipe.__dataclass_fields__ and k != "reference"})
