from .amodalvisible_evaluation import AmodalVisibleEvaluator
from .amodal_evaluation import AmodalEvaluator
from .visible_evaluation import VisibleEvaluator

# Optional text evaluation stack (pulls extra deps + older RapidFuzz APIs). Not needed for
# ECC instance segmentation baselines.
try:  # pragma: no cover
    from .text_evaluation import TextEvaluator  # noqa: F401
    from .text_eval_script import text_eval_main  # noqa: F401
    from . import rrc_evaluation_funcs  # noqa: F401
except Exception:
    pass
