import yaml
import os
import numpy as np
from sentence_transformers import SentenceTransformer

class Model:
    def __init__(self, model_path, settings, **kwargs):
        self.settings = settings
        self.settings.update(kwargs)

    # model_path ya debe ser relativo a la raíz
        self.device = "cuda" if self.settings["USE_GPU"] else "cpu"
        self.model = SentenceTransformer(model_path, device=self.device)

    def encode(self, texts, precision = np.float32, **kwargs):
        return self.model.encode(
            texts,
            device=self.device,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.settings["EMBEDDINGS_BATCH_SIZE"],
            **kwargs
        ).astype(precision)