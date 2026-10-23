from __future__ import annotations

from pathlib import Path

import numpy as np

from .media import video_frames_bgr

MODELS = Path(__file__).resolve().parent.parent / ".runtime/ComfyUI/models/freefield_eval"
YUNET = MODELS / "face_detection_yunet_2023mar.onnx"
SUBJECT_MIN = 0.62
DESCRIPTION_MIN = 0.03
MOTION_MIN = 0.05
HYBRID_MAX = 0.02
HYBRID_TEXT = "a photo of a person with animal features: cat ears, whiskers, fur on the face or an animal nose"
HUMAN_TEXT = "a photo of an ordinary person with a normal human face"


class Faces:

    ARCFACE_5PTS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

    def __init__(self) -> None:
        import cv2
        import onnxruntime as ort

        self.cv2 = cv2
        self.det = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.6)
        self.sess = ort.InferenceSession(str(MODELS / "auraface_glintr100.onnx"), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name

    def embed(self, path: Path) -> np.ndarray | None:
        return self.embed_bgr(self.cv2.imread(str(path)))

    def embed_bgr(self, bgr: np.ndarray) -> np.ndarray | None:
        cv2 = self.cv2
        h, w = bgr.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(bgr)
        if faces is None or len(faces) == 0:
            return None
        embs = []
        for face in faces:
            m, _ = cv2.estimateAffinePartial2D(face[4:14].reshape(5, 2).astype(np.float32), self.ARCFACE_5PTS, method=cv2.LMEDS)
            crop = cv2.warpAffine(bgr, m, (112, 112))
            x = ((cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
            e = self.sess.run(None, {self.input: x})[0][0]
            embs.append(e / np.linalg.norm(e))
        return np.stack(embs)

    def similarity(self, photo: Path, image: Path) -> float | None:
        ea, eb = self.embed(photo), self.embed(image)
        return None if ea is None or eb is None else float((eb @ ea[0]).max())

    def end_identity(self, photo: Path, clip: Path, last: int = 12) -> float | None:
        ea = self.embed(photo)
        if ea is None:
            return None
        frames = video_frames_bgr(clip)
        if not frames:
            return None
        scores = []
        for f in frames[-last:]:
            e = self.embed_bgr(f)
            scores.append(0.0 if e is None else float((e @ ea[0]).max()))
        return float(np.median(scores))


class FacialHair:

    TEXTS = ["a photo of a man with a moustache", "a photo of a clean-shaven man"]

    def __init__(self, faces: "Faces") -> None:
        import open_clip
        import torch

        self.torch, self.faces = torch, faces
        self.model, _, self.prep = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(MODELS / "open_clip_vit_b32_laion2b.bin"))
        self.model.eval()
        with torch.no_grad():
            te = self.model.encode_text(open_clip.get_tokenizer("ViT-B-32")(self.TEXTS))
        self.te = te / te.norm(dim=-1, keepdim=True)

    def face_crop(self, bgr: np.ndarray, up: int = 1) -> np.ndarray | None:
        import cv2

        if up > 1:
            bgr = cv2.resize(bgr, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        h, w = bgr.shape[:2]
        self.faces.det.setInputSize((w, h))
        _, found = self.faces.det.detect(bgr)
        if found is None or len(found) == 0:
            return None
        x, y, fw, fh = max(found, key=lambda f: f[2] * f[3])[:4]
        cx, cy, r = x + fw / 2, y + fh / 2, 0.8 * max(fw, fh)
        return bgr[max(0, int(cy - r)):int(cy + r), max(0, int(cx - r)):int(cx + r)]

    def p_moustache(self, bgr: np.ndarray | None) -> float | None:
        import cv2
        from PIL import Image

        if bgr is None or bgr.size == 0:
            return None
        with self.torch.no_grad():
            ie = self.model.encode_image(self.prep(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0))
            ie = ie / ie.norm(dim=-1, keepdim=True)
            return float((100 * ie @ self.te.T).softmax(-1)[0][0])

    def matches(self, photo: Path, start: Path, hole: tuple[int, int, int, int]) -> tuple[float | None, float | None]:
        import cv2

        x, y, w, h = hole
        painting = cv2.imread(str(start))[y:y + h, x:x + w]
        return self.p_moustache(self.face_crop(cv2.imread(str(photo)))), self.p_moustache(self.face_crop(painting, up=3))


class Subject:
    def __init__(self) -> None:
        import open_clip
        import torch

        self.torch = torch
        self.model, _, self.prep = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(MODELS / "open_clip_vit_b32_laion2b.bin"))
        self.model.eval()

    def embed(self, path: Path):
        from PIL import Image

        with self.torch.no_grad():
            e = self.model.encode_image(self.prep(Image.open(path).convert("RGB"))[None])
        return e / e.norm()

    def similarity(self, a: Path, b: Path) -> float:
        return float((self.embed(a) @ self.embed(b).T).item())

    def text_similarity(self, text: str, path: Path) -> float:
        import open_clip

        with self.torch.no_grad():
            t = self.model.encode_text(open_clip.get_tokenizer("ViT-B-32")([text]))
        return float((self.embed(path) @ (t / t.norm()).T).item())


def motion(clip: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(clip))
    first = last = None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        small = cv2.resize(f, (120, 208)).astype(np.float32)
        first = small if first is None else first
        last = small
    cap.release()
    return 0.0 if first is None else float(np.mean(np.abs(last - first))) / 255
