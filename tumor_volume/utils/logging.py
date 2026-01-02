# from __future__ import annotations

import subprocess
from typing import Any

import mlflow
from omegaconf import DictConfig


def _get_git_commit_hash() -> str:
    """
    Возвращает текущий git commit hash.
    Если репозиторий не git (например, в docker image),
    возвращает 'unknown'.
    """
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def setup_mlflow(cfg: DictConfig) -> None:
    """
    Инициализация MLflow.

    Ожидаемый конфиг (Hydra):
    logging:
      uri: http://127.0.0.1:8080
      experiment_name: pet_tumor_segmentation
      run_name: baseline_unet3d   # опционально
    """

    mlflow.set_tracking_uri(cfg.uri)
    mlflow.set_experiment(cfg.experiment_name)

    run_kwargs: dict[str, Any] = {}
    if "run_name" in cfg:
        run_kwargs["run_name"] = cfg.run_name

    mlflow.start_run(**run_kwargs)

    # логируем git commit
    mlflow.set_tag("git_commit", _get_git_commit_hash())

    # логируем все гиперпараметры (flattened)
    mlflow.log_params(_flatten_dict(cfg))

def _flatten_dict(
    cfg: DictConfig,
    parent_key: str | None = None,
    sep: str = ".",
) -> dict[str, Any]:
    """
    Превращает вложенный Hydra-конфиг в плоский словарь
    для MLflow.

    training.lr → 1e-4
    model.num_classes → 2
    """
    items: dict[str, Any] = {}

    for key, value in cfg.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, DictConfig):
            items.update(_flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value

    return items