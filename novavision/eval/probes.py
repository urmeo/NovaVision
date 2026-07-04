"""Affect recovery probes.

A probe reads an emotion (and graded valence/arousal) back from an image. The
benchmark's validity rests on the probe being independent of the conditioning,
so probes are swappable behind one interface and the headline result is checked
across more than one of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image

from novavision.taxonomy import (
    AROUSAL_LADDER,
    EMOTION_PROMPTS,
    EMOTIONS,
    VALENCE_LADDER,
    prior,
)

# Emotion is an argmax, so the full CLIP temperature is right. Valence/arousal
# are an expected value over a ladder, where that temperature would collapse the
# readout to one anchor; a gentler temperature keeps it graded.
_VA_TEMPERATURE = 10.0


@dataclass(frozen=True)
class Recovery:
    emotion: str
    valence: float
    arousal: float
    scores: dict[str, float]


class Probe(ABC):
    """Reads affect back from a generated image."""

    name = "base"

    @abstractmethod
    def recover(self, image: Image.Image) -> Recovery: ...

    def clip_t(self, image: Image.Image, text: str) -> float:
        return float("nan")  # undefined off CLIP


class HFImageClassifierProbe(Probe):
    """A non-CLIP image-emotion classifier, independent of the prompt vocabulary.

    Wraps a HuggingFace image-classification model. Its labels are mapped to the
    Ekman set (identity if already Ekman); valence/arousal fall back to the
    recovered emotion's prior, so this probe's strength is independent emotion
    recovery, not graded affect. Pair with ``eval.validate_probe`` to report its
    known error before trusting it.
    """

    def __init__(
        self,
        model_id: str,
        label_map: dict | None = None,
        device: str | None = None,
        revision: str | None = None,
    ):
        self.model_id = model_id
        self.label_map = {k.lower(): v for k, v in (label_map or {}).items()}
        self.revision = revision
        self.device = device
        self.name = f"img:{model_id.rsplit('/', 1)[-1]}"
        self._pipe: Any = None
        self._top_k: int | None = None

    def _load(self):
        if self._pipe is None:
            from transformers import pipeline

            pipe = pipeline(
                "image-classification",
                model=self.model_id,
                revision=self.revision,
                device=0 if self.device == "cuda" else -1,
            )
            # Assign only after the coverage check, so a failed load stays retryable.
            self._check_label_coverage(pipe.model.config.id2label)
            # transformers drops top_k=None and defaults to top 5, which could leave
            # zero mappable labels for one image; ask for every label explicitly.
            self._top_k = int(pipe.model.config.num_labels)
            self._pipe = pipe

    def _check_label_coverage(self, id2label: Mapping) -> list[str]:
        """Fail at load time if no model label reaches the Ekman set."""
        mapped = sorted(
            {self.label_map.get(str(v).lower(), str(v).lower()) for v in id2label.values()}
            & set(EMOTIONS)
        )
        if not mapped:
            raise ValueError(
                f"{self.model_id}: none of its labels map to the Ekman set; pass label_map"
            )
        return mapped

    def recover(self, image: Image.Image) -> Recovery:
        self._load()
        known = set(EMOTIONS)
        scores: dict[str, float] = {}
        for pred in self._pipe(image, top_k=self._top_k):
            label = pred["label"].lower()
            label = self.label_map.get(label, label)
            if label in known:
                scores[label] = scores.get(label, 0.0) + float(pred["score"])
        if not scores:
            # A silent `neutral` here would be indistinguishable from a genuine
            # neutral prediction and inflate the collapse diagnostic.
            raise RuntimeError(f"{self.model_id} produced no Ekman-mappable labels for this image")
        emotion = max(scores, key=lambda k: scores[k])
        v, a = prior(emotion)
        return Recovery(emotion, v, a, scores)


def _softmax(values):
    import numpy as np

    z = np.asarray(values, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class CLIPProbe(Probe):
    """Zero-shot emotion and graded valence/arousal with CLIP.

    Logits are scaled by the model's learned temperature before softmax, so the
    recovered valence/arousal span a usable range instead of being squeezed
    toward zero. Valence/arousal are read as the expected value over an ordered
    ladder of anchor prompts, not a single positive/negative contrast.
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | None = None,
        revision: str | None = None,
    ):
        self.model_id = model_id
        self.revision = revision
        self.name = f"clip:{model_id.rsplit('/', 1)[-1]}"
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._scale: float = 1.0
        self._emo_feats: Any = None
        self._val_feats: Any = None
        self._aro_feats: Any = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = (
                CLIPModel.from_pretrained(self.model_id, revision=self.revision)
                .to(self._device)
                .eval()
            )
            self._processor = CLIPProcessor.from_pretrained(self.model_id, revision=self.revision)
            self._scale = float(self._model.logit_scale.exp().item())

    def _image_features(self, image: Image.Image):
        import torch

        self._load()
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            pooled = self._model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            feats = self._model.visual_projection(pooled)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _text_features(self, texts):
        import torch

        self._load()
        inputs = self._processor(text=list(texts), return_tensors="pt", padding=True).to(
            self._device
        )
        with torch.no_grad():
            pooled = self._model.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            ).pooler_output
            feats = self._model.text_projection(pooled)
        return feats / feats.norm(dim=-1, keepdim=True)

    def _fixed_features(self):
        if self._emo_feats is None:
            import torch

            means = []
            for label in EMOTIONS:  # canonical order, not dict order
                feats = self._text_features(EMOTION_PROMPTS[label])  # ensemble
                mean = feats.mean(dim=0)
                means.append(mean / mean.norm())
            self._emo_feats = torch.stack(means)
            self._val_feats = self._text_features([p for p, _ in VALENCE_LADDER])
            self._aro_feats = self._text_features([p for p, _ in AROUSAL_LADDER])
        return self._emo_feats, self._val_feats, self._aro_feats

    def _expected(self, img, feats, ladder):
        import numpy as np

        sims = (img @ feats.T).squeeze(0).cpu().numpy()
        probs = _softmax(_VA_TEMPERATURE * sims)
        return float(np.dot(probs, [v for _, v in ladder]))

    def recover(self, image: Image.Image) -> Recovery:
        labels = list(EMOTIONS)  # must match _fixed_features stacking order
        emo_feats, val_feats, aro_feats = self._fixed_features()
        img = self._image_features(image)

        emo = (img @ emo_feats.T).squeeze(0).cpu().numpy()
        probs = _softmax(self._scale * emo)
        scores = {label: float(p) for label, p in zip(labels, probs)}
        emotion = labels[int(probs.argmax())]

        valence = self._expected(img, val_feats, VALENCE_LADDER)
        arousal = self._expected(img, aro_feats, AROUSAL_LADDER)
        return Recovery(emotion, valence, arousal, scores)

    def clip_t(self, image: Image.Image, text: str) -> float:
        img = self._image_features(image)
        txt = self._text_features([text])
        return float((img @ txt.T).item())
