import sys
from pathlib import Path

# src/ dipakai sebagai namespace package biasa (tanpa __init__.py), jadi
# root proyek harus ada di sys.path supaya "from src.config import ..."
# di modul-modul api/ bisa di-resolve, apa pun cara uvicorn dijalankan.
_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
