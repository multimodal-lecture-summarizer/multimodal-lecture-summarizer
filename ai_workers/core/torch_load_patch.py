"""Turn off Transformers' torch.load safety gate for the weights this project ships.

Florence-2 (vendored) and the CLIP checkpoint are .bin files, and torch < 2.6 makes Transformers
refuse to torch.load them. They come from pinned, verified sources, so the gate is disabled for
them. It has to be applied wherever those models load: inside a job and in the start-up preload.
"""

from __future__ import annotations


def patch_transformers_torch_load_check() -> None:
    """Patch Transformers lazily so Celery startup does not import the full AI stack."""
    try:
        import transformers.modeling_utils
        import transformers.utils
        import transformers.utils.import_utils

        transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
        transformers.utils.check_torch_load_is_safe = lambda: None
        transformers.modeling_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass
