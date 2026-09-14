import pickle
import os

MODELS_PATH = "models"

DEFAULT_POPULAR = [461684, 257040, 381170, 119736, 213834, 320130, 445351, 7943, 420960, 152913]

class ColdStartHandler:
    def __init__(self, models_path: str = MODELS_PATH):
        pop_path = os.path.join(models_path, "popularity_baseline.pkl")
        if os.path.exists(pop_path):
            with open(pop_path, "rb") as f:
                self.top_popular = pickle.load(f)
        else:
            self.top_popular = DEFAULT_POPULAR

    def get_fallback_recommendations(self, k: int = 10):
        """Returns the top-k globally popular items for unknown/cold users."""
        items = self.top_popular if self.top_popular else DEFAULT_POPULAR
        return items[:k]

