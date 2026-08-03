"""Inference-model ports — one per stage.

The application depends on these abstractions; concrete models implement them in
:mod:`adapters.outbound`:

* :class:`AnomalyModel` (stage-one) — the PatchCore ONNX + FAISS adapter, or any
  future detector. Scores **one** image at a time.
* :class:`ClusterModel` (stage-two) — the ResNet50 / DINOv2 batch clusterers, or
  any future clustering model. Groups a **whole set** of defect images at once.

They live together because they play the same role — the model the pipeline runs
on an image — and differ only in the stage they serve and their granularity.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence, Tuple

from domain.models import ClusterAssignment, ClusterFit


class AnomalyModel(ABC):
    """An anomaly-detection model that scores a single image."""

    @abstractmethod
    def inference(self, image: Any, transform: Any = None) -> Tuple[Any, Any, float]:
        """Score one image.

        Returns ``(original_img_np, dist_score, anomaly_score)``:

        * ``original_img_np`` — the preprocessed image tensor (for visualisation).
        * ``dist_score``      — per-patch L2 distance map.
        * ``anomaly_score``   — the image-level score (max patch distance).

        ``image`` may be a path or any file-like object the adapter can open.
        """
        ...

    def inference_batch(
        self, images: Sequence[Any], transform: Any = None
    ) -> list[Tuple[Any, Any, float]]:
        """Score several images in one call, returning one triple per image.

        The batched entry point the Spark scorer uses: it lets an adapter feed a
        whole mini-batch through its backbone at once instead of paying the
        per-call overhead ``inference`` does image by image. Results are aligned
        to ``images``.

        Not abstract — this default keeps every existing
        :class:`AnomalyModel` working by looping. Adapters whose backbone accepts
        a batch axis override it (see
        :meth:`adapters.outbound.patchcore.model.PatchCore.inference_batch`).
        """
        return [self.inference(image, transform) for image in images]


class ClusterModel(ABC):
    """A clustering model with two modes: fit the whole set, or assign to it.

    Stage-two clustering is **whole-set** — the experiments fit UMAP/HDBSCAN over
    the entire defect set at once (plus a KMeans second-layer split), so there is
    no per-image ``predict``. But re-fitting on every batch makes ``cluster_id``
    meaningless across runs, so the port separates the two:

    * :meth:`fit`    — cluster a whole set *and* hand back the fitted state.
    * :meth:`assign` — project a new batch through a previously fitted state,
      giving it ids from those same clusters.

    An adapter that cannot be frozen implements only :meth:`fit` and returns no
    state; the caller then has to re-fit every time.
    """

    @abstractmethod
    def fit(
        self, defect_images: Sequence[bytes], normal_images: Sequence[bytes]
    ) -> ClusterFit:
        """Cluster ``defect_images`` as a whole set, and freeze the fit.

        ``normal_images`` are recent known-good images used to build the normal
        texture baseline the descriptors are measured against. Returns a
        :class:`~domain.models.ClusterFit` whose ``assignments`` are one per
        defect image, **aligned to the input order** (``cluster_id`` ``-1`` means
        HDBSCAN noise), and whose ``state`` is the serialised bundle
        :meth:`assign` needs later.
        """
        ...

    def assign(self, defect_images: Sequence[bytes]) -> list[ClusterAssignment]:
        """Assign new images to the clusters of the state this model was built with.

        Returns one assignment per image, aligned to the input order, using the
        *same* ``cluster_id`` space as the fit that produced the state — that is
        the whole point: ids stay comparable, so a defect type keeps its number.
        ``-1`` still means "matched no cluster", and a rising share of it is the
        signal that a new defect type has appeared and a re-fit is due.

        Not abstract: adapters whose pipeline cannot be frozen simply do not
        support it, and the caller re-fits instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot assign to a frozen fit; re-fit instead"
        )
