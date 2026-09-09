import hashlib
import re
from typing import List

import numpy as np


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_\-\.]+", text.lower())


def encode_instruction(text: str, max_len: int = 16, vocab_size: int = 4096) -> np.ndarray:
    tokens = _tokenize(text)
    ids = np.zeros((max_len,), dtype=np.int64)
    for i, token in enumerate(tokens[:max_len]):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        ids[i] = (int(digest[:8], 16) % (vocab_size - 1)) + 1
    return ids

