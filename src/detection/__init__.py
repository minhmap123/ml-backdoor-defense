from .base import BaseDetector
from .mlbd import MLBDDetector
from .mlbd_cso import MLBDCSODetector
from .mm_bd import MMBDDetector
from .mmbd_cso import MMBDCSODetector
from .nc_cso import NCCSODetector
from .neural_cleanse import NeuralCleanseDetector
from .pt_red import PTREDDetector
from .pt_red_cso import PTREDCSODetector
from .types import ArtifactIndex, DetectorContext, DetectorResult, FeatureMetadata


class NoOpDetector(BaseDetector):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.requires_model = False
        self.requires_detection_split = False

    def _run_impl(self, context: DetectorContext) -> DetectorResult:
        return DetectorResult(
            detector_name=self.name,
            track_type="none",
            status="skipped",
            seed=int(context.seed),
            runtime_sec=0.0,
            summary_metrics={"detection/skipped": 1.0},
            predicted_is_infected=None,
            predicted_target_class=None,
            deviation_note="No detection executed.",
        )


DETECTION_REGISTRY = {
    "none": NoOpDetector,
    "mlbd": MLBDDetector,
    "mlbd_cso": MLBDCSODetector,
    "nc_cso": NCCSODetector,
    "mm_bd": MMBDDetector,
    "mmbd_cso": MMBDCSODetector,
    "neural_cleanse": NeuralCleanseDetector,
    "neural_cleanse_cso": NCCSODetector,
    "pt_red": PTREDDetector,
    "pt_red_cso": PTREDCSODetector,
}


def get_detection(cfg, detection_name=None):
    name = str(detection_name or cfg.name).lower()
    try:
        return DETECTION_REGISTRY[name](cfg)
    except KeyError as exc:
        available = ", ".join(sorted(DETECTION_REGISTRY))
        raise ValueError(f"Unknown detection: {name}. Available detection methods: {available}") from exc

__all__ = [
    "ArtifactIndex",
    "BaseDetector",
    "DETECTION_REGISTRY",
    "DetectorContext",
    "DetectorResult",
    "FeatureMetadata",
    "MLBDDetector",
    "MLBDCSODetector",
    "NCCSODetector",
    "MMBDDetector",
    "MMBDCSODetector",
    "NeuralCleanseDetector",
    "NoOpDetector",
    "PTREDDetector",
    "PTREDCSODetector",
    "get_detection",
]
