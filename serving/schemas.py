from pydantic import BaseModel
from typing import List

class RecommendationItem(BaseModel):
    item_id: int
    score: float
    rank: int

class RecommendationResponse(BaseModel):
    user_id: int
    is_cold_start: bool
    recommendations: List[RecommendationItem]
    latency_ms: float
