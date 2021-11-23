""" main.py
    Main entry-point for the Ride main wrapper.
    For logging to be formatted consistently, this file should be imported prior to other libraries

   isort:skip_file
"""

# Monkey-patch logger to enforce consistent look across libraries
import logging

original_getLogger = logging.getLogger


def patched_getLogger(name: str = None):
    if name:
        name = name.split(".")[0]  # Get chars before '.'
    if name == "pytorch_lightning":
        name = "lightning"
    return original_getLogger(name)


logging.getLogger = patched_getLogger

from ride.utils.logging import getLogger, init_logging, style_logging  # noqa: E402

style_logging()


from argparse import ArgumentParser  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Type, List  # noqa: E402
import yaml  # noqa: E402
import platform  # noqa: E402
from pytorch_lightning import Trainer, seed_everything  # noqa: E402
from ride.core import Configs, RideModule  # noqa: E402
from ride.hparamsearch import Hparamsearch  # noqa: E402
from ride.logging import experiment_logger  # noqa: E402
from ride.runner import Runner  # noqa: E402
from ride.utils.checkpoints import find_checkpoint  # noqa: E402
from ride.utils import env  # noqa: E402
from ride.utils.io import bump_version, dump_yaml  # noqa: E402
from ride.utils.utils import attributedict, to_dict  # noqa: E402
from pytorch_lightning.utilities.parsing import AttributeDict  # noqa: E402

logger = getLogger(__name__)


logger.debug(yaml.dump({"Environment": {n: str(getattr(env, n)) for n in env.__all__}}))


class Main:
    """Complete main programme for the lifecycle of a machine learning project

    Usage:
        Main(YourRideModule).argparse()
    """

    def __init__(self, Module: Type[RideModule]):
        self.Module = Module
        self.module_name = Module.__name__
        self.runner = Runner(self.Module)
        self.hparamsearch = Hparamsearch(self.Module)

    def argparse(
        self,
        args: List[str] = None,
        run=True,
    ):
        parser = ArgumentParser(add_help=True)

        # Top level commands
        top_level_commands_and_descriptions = [
            (
                "hparamsearch",
                "Run hyperparameter search. The best hyperparameters will be used for subsequent lifecycle methods",
            ),
            ("train", "Run model training"),
            ("validate", "Run model evaluation on validation set"),
            ("test", "Run model evaluation on test set"),
            ("profile_model", "Profile the model"),
        ]
        prog_flow_parser = parser.add_argument_group(
            "Flow",
            description="Commands that control the top-level flow of the programme.",
        )
        for c, d in top_level_commands_and_descriptions:
            prog_flow_parser.add_argument(f"--{c}", action="store_true", help=d)

        # General settings
        gen_settings_parser = parser.add_argument_group(
            "General",
            description="Settings that apply to the programme in general.",
        )
        gen_configs = Configs()
        gen_configs.add(
            "id",
            type=str,
            default="unnamed",
            description="Identifier for the run. If not specified, the current timestamp will be used",
        )
        gen_configs.add(
            "seed",
            type=int,
            default=123,
            description="Global random seed",
        )
        gen_configs.add(
            "logging_backend",
            type=str,
            choices=("tensorboard", "wandb"),
            default="tensorboard",
            description="Type of experiment logger",
        )
        gen_configs.add(
            "from_hparams_file",
            type=str,
            default=None,
            description="Path to JSON hparams file",
        )
        gen_configs.add(
            name="optimization_metric",
            default="loss",
            type=str,
            choices=self.Module.metric_names(),
            description="Name of the performance metric that should be optimized",
        )
        gen_configs.add(
            "monitor_lr",
            type=int,
            default=1,
            choices=[0, 1],
            description="Whether to monitor and log learning rate",
        )
        gen_configs.add(
            "checkpoint_every_n_steps",
            type=int,
            default=0,
            description="Save models checkpoint every N steps independent of epoch and validation cycle. If `0`, this feature is unused",
        )
        gen_configs.add(
            "profile_model_num_runs",
            type=int,
            default=0,
            description="Number of runs to perform when profiling model. If `0`, model will be profiled to max 10 seconds.",
        )
        gen_settings_parser = gen_configs.add_argparse_args(gen_settings_parser)

        # Pytorch Lightning args
        pl_parser = parser.add_argument_group(
            "Pytorch Lightning",
            description="Settings inherited from the pytorch_lightning.Trainer",
        )
        pl_configs = Configs.from_argument_parser(
            Trainer.add_argparse_args(ArgumentParser(add_help=False))
        )
        pl_parser = pl_configs.add_argparse_args(pl_parser)

        # Hparamsearch
        hparamsearch_parser = parser.add_argument_group(
            "Hparamsearch",
            description="Settings associated with hyperparameter optimisation",
        )
        hparamsearch_parser = (
            self.hparamsearch.configs() - gen_configs
        ).add_argparse_args(hparamsearch_parser)

        # Module args
        module_parser = parser.add_argument_group(
            "Module",
            description="Settings associated with the Module",
        )
        module_parser = (
            self.Module.configs() - gen_configs - pl_configs
        ).add_argparse_args(module_parser)

        if run:
            parsed_args = parser.parse_args(args)
            return self.main(parsed_args)

        return parser

    def main(self, args: AttributeDict):  # noqa: C901
        args = attributedict(args)

        # Ensure gpus is defined
        args.gpus = getattr(args, "gpus", "")
        args.max_epochs = getattr(args, "max_epochs", 1)

        seed_everything(args.seed)

        self.log_dir = experiment_logger(
            args.id, args.logging_backend, getattr(self.Module, "__name__")
        ).log_dir

        init_logging(
            self.log_dir,
            args.logging_backend,
        )

        if getattr(args, "num_workers", False) and platform.system() == "Windows":
            logger.warning(
                f"You have requested num_workers={args.num_workers} on Windows, but currently 0 is recommended (see https://stackoverflow.com/a/59680818)"
            )

        results = []

        if not args.default_root_dir:
            args.default_root_dir = str(env.LOGS_PATH)

        save_results = make_save_results(self.log_dir)

        if args.resume_from_checkpoint:
            args.resume_from_checkpoint = find_checkpoint(args.resume_from_checkpoint)

        if args.from_hparams_file:
            logger.info(f"Loading hyperparameters from {args.from_hparams_file}")
            args = Hparamsearch.load(
                args.from_hparams_file,
                old_args=args,
                Cls=self.Module,
                auto_scale_lr=True,
            )

        save_results("hparams.yaml", to_dict(args))

        if args.hparamsearch:
            hprint("Searching for optimal model hyperparameters")
            best_hparams = self.hparamsearch.run(args)
            if not best_hparams:
                raise RuntimeError("Hparamsearch was unable to identify best hparams")
            logger.info("Hparamsearch completed")
            dprint(best_hparams)
            results.append(best_hparams)
            # Save to both run_logs and tune_logs
            save_results("hparamsearch.yaml", to_dict(best_hparams))
            best_hparams_path = Hparamsearch.dump(best_hparams, identifier=args.id)
            logger.info("🔧 Assigning best hparams to model")
            args = Hparamsearch.load(best_hparams_path, old_args=args)

        if args.train:
            hprint("Running training")
            self.runner.train(args)

        if args.validate:
            hprint("Running evaluation on validation set")
            val_results = self.runner.validate(args)
            dprint(val_results)
            results.append(val_results)
            save_results("val_results.yaml", val_results)

        if args.test:
            hprint(
                f"Running evaluation on test set{' using ensemble testing' if args.test_ensemble else ''}"
            )
            test_results = self.runner.test(args)
            dprint(test_results)
            results.append(test_results)
            save_results("test_results.yaml", test_results)

        if args.profile_model:
            hprint("Profiling model")
            info = self.runner.profile_model(args, num_runs=args.profile_model_num_runs)
            dprint(info)
            results.append(info)
            save_results("profile.yaml", info)

        return results


def hprint(msg: str):
    """Message header print

    Args:
        msg (str): Message to be printed
    """
    logger.info(f"🚀 {msg}")


def dprint(d: dict):
    logger.info(yaml.dump({"Results": d}))


def make_save_results(root_path: str, verbose=True) -> Callable[[str, Any], None]:
    if root_path is None:  # pragma: no cover

        def dummy(*args, **kwargs):
            return None

        return dummy

    root_path = Path(root_path)

    def bump_version_and_save(relative_path: str, data):
        nonlocal root_path
        path = bump_version(root_path / relative_path)
        if verbose:
            logger.info("💾 Saving " + str(path))
        dump_yaml(path, data)
        return path

    return bump_version_and_save
