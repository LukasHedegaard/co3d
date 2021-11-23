import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

__all__ = [
    "DATASETS_PATH",
    "LOGS_PATH",
    "RUN_LOGS_PATH",
    "TUNE_LOGS_PATH",
    "CACHE_PATH",
    "LOG_LEVEL",
    "NUM_CPU",
]

DATASETS_PATH = Path(os.getenv("DATASETS_PATH", default="datasets"))
LOGS_PATH = Path(os.getenv("LOGS_PATH", default="logs"))
RUN_LOGS_PATH = LOGS_PATH / "run_logs"
TUNE_LOGS_PATH = LOGS_PATH / "tune_logs"
CACHE_PATH = Path(os.getenv("CACHE_PATH", default=".cache"))
LOG_LEVEL = os.getenv("LOG_LEVEL", default="INFO")
NUM_CPU = os.cpu_count() or 1

DATASETS_PATH.mkdir(exist_ok=True)
LOGS_PATH.mkdir(exist_ok=True)
RUN_LOGS_PATH.mkdir(exist_ok=True)
TUNE_LOGS_PATH.mkdir(exist_ok=True)
CACHE_PATH.mkdir(exist_ok=True)
