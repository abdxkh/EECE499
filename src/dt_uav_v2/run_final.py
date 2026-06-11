from pathlib import Path

from dt_uav_v2.training.train_manager import train_manager
from dt_uav_v2.training.train_worker import train_worker


def main():
    """
    First project entry point.

    If a worker checkpoint exists, train the manager on top of it. Otherwise,
    train the worker first, then train the manager.
    """

    worker_path = Path("outputs/models/worker_continuous_final.pt")

    if not worker_path.exists():
        train_worker(save_path=str(worker_path))

    train_manager(
        worker_model_path=str(worker_path),
        save_path="outputs/models/manager_backhaul_final.pt",
    )


if __name__ == "__main__":
    main()
