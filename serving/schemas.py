from pydantic import BaseModel, Field
from typing import List, Optional

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="Unique Visitor / User ID")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of items to recommend")

class RecommendationItem(BaseModel):
    item_id: int
    score: float
    category: Optional[int] = None
    category_id: Optional[int] = None
    rank: Optional[int] = None

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecommendationItem]
    fallback_used: bool = False
    is_cold_start: bool = False
    latency_ms: Optional[float] = None

class ExplainRequest(BaseModel):
    item_id: int
    score: float
    category_id: Optional[int] = None
    top_features: Optional[List[str]] = Field(default_factory=list)
    shopper_recent_categories: Optional[List[int]] = Field(default_factory=list)

class ExplainResponse(BaseModel):
    item_id: int
    explanation: str


