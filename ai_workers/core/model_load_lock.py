"""Process-wide lock for building transformers models.

`from_pretrained` temporarily patches torch/transformers globals (empty-weights init), so two
threads loading models at once can leave parameters on the meta device
("Cannot copy out of meta tensor"). The audio and visual branches run concurrently, so every
transformers model load must hold this lock. Inference itself does not need it.
"""

from __future__ import annotations

import threading

MODEL_LOAD_LOCK = threading.RLock()
