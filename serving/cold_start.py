import pickle
import os

MODELS_PATH = "models"

class ColdStartHandler:
    def __init__(self):
        pop_path = os.path.join(MODELS_PATH, "popularity_baseline.pkl")
        if os.path.exists(pop_path):
            with open(pop_path, "rb") as f:
                self.top_popular = pickle.load(f)
        else:
            self.top_popular = []

    def get_fallback_recommendations(self, k: int = 10):
        """Returns the top-k globally popular items for unknown/cold users."""
        return self.top_popular[:k]
