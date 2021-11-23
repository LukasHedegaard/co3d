import sys
from functools import partial
from pathlib import Path
from typing import Type, Union

from pytorch_lightning.utilities.parsing import AttributeDict

from ride.core import Configs, RideModule
from ride.runner import Runner, is_runnable
from ride.utils.env import NUM_CPU, TUNE_LOGS_PATH
from ride.utils.gpus import parse_num_gpus
from ride.utils.io import bump_version, dump_json, dump_yaml, load_structured_data
from ride.utils.logging import getLogger

logger = getLogger(__name__)


class Hparamsearch:
    def __init__(self, Module: Type[RideModule]):
        assert is_runnable(Module)
        self.Module = Module
        self.module_name = Module.__name__

    def configs(self) -> Configs:
        c = Configs()
        c.add(
            name="trials",
            default=30,
            type=int,
            description="Number of trials in the hyperparameter search",
        )
        c.add(
            name="gpus_per_trial",
            default=0,
            type=int,
            description="Number of GPUs per trail in the hyperparameter search",
        )
        c.add(
            name="optimization_metric",
            default="loss",
            type=str,
            choices=self.Module.metric_names(),
            description="Name of the performance metric that should be optimized",
        )
        c.add(
            name="from_hparam_space_file",
            default=None,
            type=str,
            description="Path to file with specification for the search space during hyper-parameter optimisation",
        )
        return c

    def __call__(self, args: AttributeDict):
        self.run(args)

    def run(self, args: AttributeDict):
        """Run hyperparameter search using the `tune.schedulers.ASHAScheduler`

        Args:
            args (AttributeDict): Arguments

        Side-effects:
            Saves logs to `TUNE_LOGS_PATH / args.id`
        """
        try:
            from ray import tune
            from ray.tune.integration.pytorch_lightning import (
                TuneReportCheckpointCallback,
            )
        except ModuleNotFoundError as e:  # pragma: no cover
            logger.error(
                "To use hyperparameter search, first install Ray Tune via `pip install 'ray[tune]'` or `pip install 'ride[extras]'`"
            )
            raise e

        if not hasattr(args, "id"):
            args.id = "hparamsearch"

        module_config = (
            Configs.from_file(args.from_hparam_space_file)
            if args.from_hparam_space_file
            else self.Module.configs()
        ).tune_config()

        config = {
            **dict(args),
            **module_config,
            # pl.Trainer args:
            "gpus": args.gpus_per_trial,
            "logger": False,
            "accumulate_grad_batches": (
                (8 // args.gpus_per_trial) * args.accumulate_grad_batches
                if args.gpus_per_trial
                else args.accumulate_grad_batches
            ),
        }
        scheduler = tune.schedulers.ASHAScheduler(
            metric=f"val/{args.optimization_metric}",
            mode=self.Module.metrics()[args.optimization_metric].value,
            max_t=args.max_epochs,
            grace_period=1,
            reduction_factor=2,
        )

        metric_names = [f"val/{m}" for m in self.Module.metrics().keys()]

        reporter = tune.CLIReporter(
            metric_columns=[*metric_names, "training_iteration"],
        )
        tune_callbacks = [
            TuneReportCheckpointCallback(
                metrics=metric_names,
                filename="checkpoint",
                on="validation_end",
            )
        ]
        cpus_per_trial = max(
            1,
            (
                min(10 * args.gpus_per_trial, NUM_CPU - 10)
                if args.gpus_per_trial
                else min(10, NUM_CPU - 2)
            ),
        )

        analysis = tune.run(
            partial(
                Runner.static_train_and_val,
                self.Module,
                trainer_callbacks=tune_callbacks,
            ),
            name=args.id,
            local_dir=str(TUNE_LOGS_PATH),
            resources_per_trial={"cpu": cpus_per_trial, "gpu": args.gpus_per_trial},
            config=config,
            num_samples=args.trials,
            scheduler=scheduler,
            progress_reporter=reporter,
            queue_trials=False,
            raise_on_failed_trial=False,
        )

        best_hparams = analysis.get_best_config(
            metric=f"val/{args.optimization_metric}",
            mode=self.Module.metrics()[args.optimization_metric].value,
            scope="all",
        )
        # Select only model parameters
        if best_hparams:
            best_hparams = {
                k: best_hparams[k]
                for k in [
                    *self.Module.configs().names,
                    # Trainer parameters that influence model hparams:
                    "accumulate_grad_batches",
                    "batch_size",
                    "gpus",
                ]
            }
        return best_hparams

    @staticmethod
    def dump(hparams: dict, identifier: str, extention="yaml") -> str:
        """Dumps haparams to TUNE_LOGS_PATH / identifier / "best_hparams.json" """
        dump_path = bump_version(
            TUNE_LOGS_PATH / identifier / f"best_hparams.{extention}"
        )
        dump = {"json": dump_json, "yaml": dump_yaml}[extention]
        dump(dump_path, hparams)
        return str(dump_path)

    @staticmethod
    def load(
        path: Union[Path, str],
        old_args=AttributeDict(),
        Cls: Type[RideModule] = None,
        auto_scale_lr=False,
    ) -> AttributeDict:
        """Loads hparams from path

        Args:
            path (Union[Path, str]): Path to jsonfile containing hparams
            old_args (Optional[AttributeDict]):The AttributeDict to be updated with the new hparams
            cls (Optional[RideModule]): A class whole hyperparameters can be used to select the relevant hparams to take

        Returns:
            AttributeDict: AttributeDict with updated hyperparameters
        """
        path = Path(path)
        hparams = load_structured_data(path)

        if Cls:
            hparam_names = Cls.configs().names
            hparams = {k: v for k, v in hparams.items() if k in hparam_names}

        # During hparamsearch, only a single GPU is used, but accumulate_grad_batches is set to the total number of gpus given
        # If we have multiple GPUs, we need to reduce accumulate_grad_batches accordingly
        num_gpus = parse_num_gpus(old_args.gpus)
        if num_gpus > 0 and "accumulate_grad_batches" in hparams:  # pragma: no cover
            hparams["accumulate_grad_batches"] = max(
                1, int(hparams["accumulate_grad_batches"]) // num_gpus
            )

        old_args = dict(old_args)
        user_passed_arg_keys = [a[2:] for a in sys.argv if a.startswith("--")]
        user_passed_args = {
            k: v for k, v in old_args.items() if k in user_passed_arg_keys
        }

        # If batch size was changed by user, automatically apply the linear scaling rule to the learning rate
        if (
            auto_scale_lr
            and "batch_size" in hparams
            and "learning_rate" in hparams
            and "batch_size" in user_passed_args
            and "learning_rate" not in user_passed_args
        ):
            old_accumulate_grad_batches = (
                hparams["accumulate_grad_batches"]
                if "accumulate_grad_batches" in hparams
                else 1
            )
            new_accumulate_grad_batches = (
                user_passed_args["accumulate_grad_batches"]
                if "accumulate_grad_batches" in user_passed_args
                else old_accumulate_grad_batches
            )
            new_tot_batch = new_accumulate_grad_batches * user_passed_args["batch_size"]
            old_tot_batch = old_accumulate_grad_batches * hparams["batch_size"]
            if new_tot_batch != old_tot_batch:
                scaling = new_tot_batch / old_tot_batch
                user_passed_args["learning_rate"] = hparams["learning_rate"] * scaling
                logger.info(
                    f"🔧 A `batch_size*accumulate_grad_batches` ({new_tot_batch}) differs from hparams file ({old_tot_batch}). "
                    f"Scaling learning_rate from {hparams['learning_rate']} to {user_passed_args['learning_rate']} (= {hparams['learning_rate']} × {new_tot_batch} / {old_tot_batch})"
                )

        return AttributeDict(**{**old_args, **hparams, **user_passed_args})
