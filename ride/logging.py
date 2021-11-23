import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pytorch_lightning as pl
from matplotlib.figure import Figure
from PIL import Image
from pytorch_lightning.loggers import (
    LightningLoggerBase,
    LoggerCollection,
    NeptuneLogger,
    TensorBoardLogger,
    WandbLogger,
)
from pytorch_lightning.utilities import rank_zero_only

from ride.metrics import FigureDict
from ride.utils.env import RUN_LOGS_PATH
from ride.utils.logging import getLogger, process_rank

logger = getLogger(__name__)
ExperimentLogger = Union[TensorBoardLogger, LoggerCollection, WandbLogger]
ExperimentLoggerCreator = Callable[[str], ExperimentLogger]


def singleton_experiment_logger() -> ExperimentLoggerCreator:
    _loggers = {}

    def experiment_logger(
        name: str = None,
        logging_backend: str = "tensorboard",
        project_name: str = None,
        save_dir=RUN_LOGS_PATH,
    ) -> ExperimentLogger:
        nonlocal _loggers
        if logging_backend not in _loggers:
            if process_rank != 0:  # pragma: no cover
                _loggers[logging_backend] = pl.loggers.base.DummyLogger()
                _loggers[logging_backend].log_dir = None
                return _loggers[logging_backend]

            logging_backend = logging_backend.lower()
            if logging_backend == "tensorboard":
                _loggers[logging_backend] = TensorBoardLogger(
                    save_dir=save_dir, name=name
                )
            elif logging_backend == "wandb":
                _loggers[logging_backend] = WandbLogger(
                    save_dir=save_dir,
                    name=name,
                    project=project_name,
                )
                _loggers[logging_backend].log_dir = getattr(
                    _loggers[logging_backend].experiment._settings, "_sync_dir", None
                )
            else:
                logger.warn("No valid logger selected.")

        return _loggers[logging_backend]

    return experiment_logger


experiment_logger = singleton_experiment_logger()


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def add_experiment_logger(
    prev_logger: LightningLoggerBase, new_logger: LightningLoggerBase
) -> LoggerCollection:
    # If no logger existed previously don't do anything
    if not prev_logger:
        return None

    if isinstance(prev_logger, LoggerCollection):
        return LoggerCollection([*prev_logger._logger_iterable, new_logger])

    return LoggerCollection([prev_logger, new_logger])


def get_log_dir(module: pl.LightningModule):
    loggers = (
        module.logger if hasattr(module.logger, "__getitem__") else [module.logger]
    )
    for lgr in loggers[::-1]:  # ResultLogger would be last
        if hasattr(lgr, "log_dir"):
            return lgr.log_dir


def log_figures(module: pl.LightningModule, d: FigureDict):
    assert isinstance(module, pl.LightningModule)
    module_loggers = (
        module.logger if hasattr(module.logger, "__getitem__") else [module.logger]
    )
    image_loggers = []
    for lgr in module_loggers:
        if type(lgr) == NeptuneLogger:
            # log_image(log_name, image, step=None)
            image_loggers.append(lgr.log_image)
        elif type(lgr) == TensorBoardLogger:
            # SummaryWriter.add_figure(self, tag, figure)
            image_loggers.append(lgr.experiment.add_figure)
        elif type(lgr) == WandbLogger:
            try:
                import wandb  # noqa: F401

                wandb_log = lgr.experiment.log

                def log_figure(tag, fig):
                    im = wandb.Image(fig2img(fig), caption=tag)
                    return wandb_log({tag: im})

                image_loggers.append(log_figure)
            except ImportError:
                logger.error(
                    "Before using the WandbLogger, first install WandB using `pip install wandb`"
                )

        elif type(lgr) == ResultsLogger:
            image_loggers.append(lgr.log_figure)

    if not image_loggers:
        logger.warn(
            f"Unable to log figures {d.keys()}: No compatible logger found among {module_loggers}"
        )
        return

    for k, v in d.items():
        for log in image_loggers:
            log(k, v)


class ResultsLogger(LightningLoggerBase):
    def __init__(self, prefix="test", save_to: str = None):
        super().__init__()
        self.results = {}
        self.prefix = prefix
        self.log_dir = save_to

    def _fix_name_perfix(self, s: str, replace="test/") -> str:
        if not self.prefix:
            return s

        if s.startswith(replace):
            return f"{self.prefix}/{s[5:]}"

        return f"{self.prefix}/{s}"

    @property
    def experiment(self):
        return None

    @rank_zero_only
    def log_hyperparams(self, params):
        ...

    @rank_zero_only
    def log_metrics(self, metrics: Dict, step):
        self.results = {self._fix_name_perfix(k): float(v) for k, v in metrics.items()}

    def log_figure(self, tag: str, fig: Figure):
        if self.log_dir:
            fig_path = Path(self.log_dir) / "figures" / f"{tag}.png"
            logger.info(f"💾 Saving figure {tag} to {str(fig_path)}")
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(fig_path), bbox_inches="tight")

    @rank_zero_only
    def finalize(self, status):
        pass

    @property
    def save_dir(self) -> Optional[str]:
        return self.log_dir

    @property
    def name(self):
        return "ResultsLogger"

    @property
    def version(self):
        return "1"


StepOutputs = List[Dict[str, Any]]
