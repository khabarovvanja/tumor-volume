import subprocess
from pathlib import Path


def download_data(data_dir: str | Path | None = None) -> None:
    """
    Скачивает данные с Google Drive через DVC.

    Ожидается, что:
    - dvc remote уже настроен (gdrive)
    - .dvc файлы находятся в репозитории
    - функция вызывается из корня проекта

    Parameters
    ----------
    data_dir : str | Path | None
        Необязательный путь. Если указан — проверяется наличие данных.
    """

    if data_dir is not None:
        data_dir = Path(data_dir)
        if data_dir.exists() and any(data_dir.iterdir()):
            # данные уже есть
            return

    try:
        subprocess.run(
            ["dvc", "pull"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to download data via DVC. "
            "Make sure DVC is installed and Google Drive remote is accessible."
        ) from exc