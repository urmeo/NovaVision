"""CLIP-based affect recovery from generated images."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from novavision.taxonomy import AROUSAL_ANCHORS, EMOTION_PROMPTS, VALENCE_ANCHORS

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


@dataclass(frozen=True)
class Recovery:
    emotion: str
    valence: float
    arousal: float
    scores: dict[str, float]


def _softmax(values):
    import numpy as np

    z = np.asarray(values, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class CLIPAffect:
    """Zero-shot emotion and valence/arousal probing with CLIP."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        self.model_id = model_id
        self._device = device
        self._model = None
        self._processor = None
        self._emo_feats = None
        self._anchor_feats = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = CLIPModel.from_pretrained(self.model_id).to(self._device).eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_id)

    def _image_features(self, image: Image.Image):
        import torch

        self._load()
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _text_features(self, texts):
        import torch

        self._load()
        inputs = self._processor(text=list(texts), return_tensors="pt", padding=True).to(
            self._device
        )
        with torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _fixed_features(self):
        if self._emo_feats is None:
            import torch

            means = []
            for label in EMOTION_PROMPTS:
                feats = self._text_features(EMOTION_PROMPTS[label])  # ensemble
                mean = feats.mean(dim=0)
                means.append(mean / mean.norm())
            self._emo_feats = torch.stack(means)
            self._anchor_feats = self._text_features([*VALENCE_ANCHORS, *AROUSAL_ANCHORS])
        return self._emo_feats, self._anchor_feats

    def recover(self, image: Image.Image) -> Recovery:
        labels = list(EMOTION_PROMPTS)
        emo_feats, anchor_feats = self._fixed_features()

        img = self._image_features(image)
        emo = (img @ emo_feats.T).squeeze(0).cpu().numpy()
        anc = (img @ anchor_feats.T).squeeze(0).cpu().numpy()

        probs = _softmax(emo)
        scores = {label: float(p) for label, p in zip(labels, probs)}
        emotion = labels[int(probs.argmax())]

        v_pos, v_neg, a_high, a_low = anc
        p_pos = _softmax([v_pos, v_neg])[0]
        p_high = _softmax([a_high, a_low])[0]
        return Recovery(emotion, float(2 * p_pos - 1), float(p_high), scores)

    def clip_t(self, image: Image.Image, text: str) -> float:
        img = self._image_features(image)
        txt = self._text_features([text])
        return float((img @ txt.T).item())
