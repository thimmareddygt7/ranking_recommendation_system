import pytest
from src.pipeline import run_pipeline
from pathlib import Path
import json

def test_pipeline_execution(tmp_path):
    run_pipeline("config.yaml")
    status_file = Path("results/pipeline_status.json")
    assert status_file.exists()
    
    with open(status_file, "r") as f:
        data = json.load(f)
    assert data["status"] == "SUCCESS"
