"""Central configuration for paths and constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DEFAULT_DATA_PATH = DATA_DIR / "bank.csv"
TARGET_COLUMN = "y"
RANDOM_STATE = 42
TEST_SIZE = 0.2
