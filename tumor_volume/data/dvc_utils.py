import subprocess
from pathlib import Path

def download_data(cfg) -> None:
    """
    Download raw data via DVC if it is not present locally.
    Assumes raw.dvc is in data root and creates data/raw/
    """
    data_root = Path(cfg.root_dir)  # обычно "data/"
    raw_dvc = data_root / "raw.dvc"      # путь к raw.dvc
    raw_dir = data_root / "raw"          # папка, которую создаст DVC

    if not raw_dvc.exists():
        raise RuntimeError(f"DVC file '{raw_dvc}' does not exist. Make sure it is in your repo.")

    # Проверка, есть ли данные уже
    if raw_dir.exists() and any(raw_dir.iterdir()):
        print(f"[download_data] Data already exists in '{raw_dir}', skipping download.")
        return

    print(f"[download_data] '{raw_dir}' not found. Pulling via DVC ({raw_dvc})...")
    try:
        subprocess.run(["dvc", "pull", str(raw_dvc)], check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"DVC pull failed for {raw_dvc}. Make sure DVC is installed and remote is accessible."
        ) from e

    # Финальная проверка после загрузки
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        raise RuntimeError(f"Data for '{raw_dir}' was not downloaded correctly.")

    print(f"[download_data] Data successfully downloaded in '{raw_dir}'.")