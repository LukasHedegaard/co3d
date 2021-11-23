import logging
import os
import socket
import subprocess
import sys
from functools import wraps
from pathlib import Path

import coloredlogs
import pytorch_lightning as pl

from ride.utils.env import LOG_LEVEL
from ride.utils.utils import once

LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _process_rank():
    if pl.utilities._HOROVOD_AVAILABLE:
        import horovod.torch as hvd

        hvd.init()
        return hvd.rank()

    return pl.utilities.rank_zero_only.rank


process_rank = _process_rank()


def if_rank_zero(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        global process_rank
        if process_rank == 0:
            fn(*args, **kwargs)

    return wrapped


def getLogger(name, log_once=False):
    name = name.split(".")[0]  # Get chars before '.'
    if name not in {"wandb", "lightning", "ride", "datasets", "models"}:
        name = style(name, fg="white", bold=True)
    logger = logging.getLogger(name)
    if log_once:
        logger._log = once(logger._log)
    logger._log = if_rank_zero(logger._log)
    return logger


logger = getLogger(__name__)


_ansi_colors = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "reset": 39,
    "purple": 56,
    "bright_black": 90,
    "bright_red": 91,
    "bright_green": 92,
    "bright_yellow": 93,
    "bright_blue": 94,
    "bright_magenta": 95,
    "bright_cyan": 96,
    "bright_white": 97,
}
_ansi_reset_all = "\033[0m"


def style(  # noqa: C901
    text,
    fg=None,
    bg=None,
    bold=None,
    dim=None,
    underline=None,
    blink=None,
    reverse=None,
    reset=True,
):
    """Styles a text with ANSI styles and returns the new string.  By
    default the styling is self contained which means that at the end
    of the string a reset code is issued.  This can be prevented by
    passing ``reset=False``.

    This is a modified version of the one found in `click` https://click.palletsprojects.com/en/7.x/

    Examples::

        logger.info(style('Hello World!', fg='green'))
        logger.info(style('ATTENTION!', blink=True))
        logger.info(style('Some things', reverse=True, fg='cyan'))

    Supported color names:

    * ``black`` (might be a gray)
    * ``red``
    * ``green``
    * ``yellow`` (might be an orange)
    * ``blue``
    * ``magenta``
    * ``cyan``
    * ``white`` (might be light gray)
    * ``bright_black``
    * ``bright_red``
    * ``bright_green``
    * ``bright_yellow``
    * ``bright_blue``
    * ``bright_magenta``
    * ``bright_cyan``
    * ``bright_white``
    * ``reset`` (reset the color code only)

    :param text: the string to style with ansi codes.
    :param fg: if provided this will become the foreground color.
    :param bg: if provided this will become the background color.
    :param bold: if provided this will enable or disable bold mode.
    :param dim: if provided this will enable or disable dim mode.  This is
                badly supported.
    :param underline: if provided this will enable or disable underline.
    :param blink: if provided this will enable or disable blinking.
    :param reverse: if provided this will enable or disable inverse
                    rendering (foreground becomes background and the
                    other way round).
    :param reset: by default a reset-all code is added at the end of the
                  string which means that styles do not carry over.  This
                  can be disabled to compose styles.
    """
    bits = []
    if fg:
        try:
            bits.append("\033[{}m".format(_ansi_colors[fg]))
        except KeyError:
            raise TypeError("Unknown color '{}'".format(fg))
    if bg:
        try:
            bits.append("\033[{}m".format(_ansi_colors[bg] + 10))
        except KeyError:
            raise TypeError("Unknown color '{}'".format(bg))
    if bold is not None:
        bits.append("\033[{}m".format(1 if bold else 22))
    if dim is not None:
        bits.append("\033[{}m".format(2 if dim else 22))
    if underline is not None:
        bits.append("\033[{}m".format(4 if underline else 24))
    if blink is not None:
        bits.append("\033[{}m".format(5 if blink else 25))
    if reverse is not None:
        bits.append("\033[{}m".format(7 if reverse else 27))
    bits.append(text)
    if reset:
        bits.append(_ansi_reset_all)
    return "".join(bits)


def style_logging():
    assert LOG_LEVEL in LOG_LEVELS, f"Specified LOG_LEVEL should be one of {LOG_LEVELS}"

    # Installing coloredlogs during tests doesn't work
    if "pytest" not in sys.modules:
        coloredlogs.install(
            level=LOG_LEVEL,
            fmt="%(name)s: %(message)s",
            level_styles={
                "debug": {"color": "white", "faint": True},
                "warning": {"bold": True},
                "error": {"color": "red", "bold": True},
            },
        )

    # Block pytorch_lightning from writing directly to stdout
    lightning_logger = getattr(pl, "_logger", logging.getLogger("lightning"))
    lightning_logger.handlers = []
    lightning_logger.propagate = bool(process_rank == 0)

    # Set coloring
    lightning_logger.name = style(lightning_logger.name, fg="yellow", bold=True)

    ride_logger = logging.getLogger("ride")
    ride_logger.name = style(ride_logger.name, fg="cyan", bold=True)

    datasets_logger = logging.getLogger("datasets")
    datasets_logger.name = style(datasets_logger.name, fg="magenta", bold=True)

    models_logger = logging.getLogger("models")
    models_logger.name = style(models_logger.name, fg="green", bold=True)

    # Block matplotlib debug logger
    matplotlib_logger = logging.getLogger("matplotlib")
    matplotlib_logger.setLevel(
        logging.INFO if LOG_LEVEL == "DEBUG" else getattr(logging, LOG_LEVEL)
    )


def init_logging(logdir: str = None, logging_backend: str = "tensorboard"):
    if not logdir:
        return

    # Add root handler for redirecting run output to file
    os.makedirs(logdir, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(Path(logdir) / "run.log"))

    # Write basic environment info to logs
    logger.info(f"Running on host {style(socket.gethostname(), fg='yellow')}")

    try:
        git_repo = (
            subprocess.check_output(
                "git config --get remote.origin.url",
                shell=True,
                stderr=subprocess.STDOUT,
            )
            .decode("utf-8")
            .strip()
        )
        git_tag = (
            subprocess.check_output(
                "git rev-parse HEAD", shell=True, stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )
        git_msg = style(
            f"{git_repo.replace('.git','')}/tree/{git_tag}", fg="blue", bold=False
        )
        logger.info(f"⭐️ View project repository at {git_msg}")
    except subprocess.CalledProcessError:
        pass

    logger.info(f"Run data is saved locally at {logdir}")
    logger.info(f"Logging using {style(logging_backend.capitalize(), fg='yellow')}")
