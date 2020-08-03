import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import lavd
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

default_checkpoint = {
    "epoch": 0,
    "train": {"lr": [], "stats": {"loss": [], "perplexity": []}},
    "validation": {},
    "outdated_validation": [],
    "model": {"kind": None},
}

metrics = [
    OrderedDict(name="Loss", key="loss", order="min"),
    OrderedDict(name="Perplexity", key="perplexity", order="min"),
]

repo_path = os.path.dirname(os.path.abspath(__file__))


# By default it loads it on the CPU, because it usually doesn't need to be on the GPU as
# the whole model will be switched to the device with .to(device). Loading it on the GPU
# will occupy unnecessary GPU memory.
def load_checkpoint(path: str, cuda: bool = False) -> Dict:
    device = torch.device("cuda" if cuda else "cpu")
    return torch.load(path, map_location=device)


def log_top_checkpoints(
    logger: lavd.Logger,
    results: Dict[str, Dict],
    criterion: List[OrderedDict],
    k: int = 5,
):
    lines = []
    for name, result in results.items():
        lines.append("")
        lines.append("## {}".format(name))
        for metric in metrics:
            lines.append("")
            lines.append("### {}".format(metric["name"]))
            lines.append("")
            crit = result["stats"]
            for key in metric["key"].split("."):
                crit = crit[key]
            values = torch.tensor(crit)
            descending = metric["order"] == "max"
            sorted_metric = torch.sort(values, descending=descending)
            for i, (value, index) in enumerate(zip(*sorted_metric)):
                if i >= k:
                    break
                lines.append(
                    "{i}. {path} - {value:.5f}\n".format(
                        i=i + 1,
                        path=logger.get_file_path(
                            "model", step=result["start"] + index + 1, extension=".pt"
                        ).parent.as_posix(),
                        value=value.item(),
                    )
                )
    markdown = "\n".join(lines)
    logger.log_markdown(markdown, "best")


def log_experiment(logger: lavd.Logger, experiment: Dict):
    infos = {
        "Model": experiment["model_kind"],
        "Train Dataset": experiment["train"],
        "Validation Dataset": {
            exp["name"]: {k: v for k, v in exp.items() if k != "name"}
            for exp in experiment["validation"]
        },
    }
    logger.log_summary(infos, options=experiment["options"])


def save_checkpoint(
    logger: lavd.Logger,
    model: PreTrainedModel,
    tokeniser: PreTrainedTokenizer,
    stats: Dict,
    step: int,
):
    if not logger.disabled:
        logger.save_obj(stats, "stats", step=step)
        stats_path = logger.get_file_path("stats", step=step, extension=".pt")
        out_dir = stats_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        tokeniser.save_pretrained(out_dir)


def log_results(
    logger: lavd.Logger,
    epoch: int,
    train_result: Dict,
    validation_results: List[OrderedDict],
    model: nn.Module,
):
    logger.log_scalar(train_result["lr"], "learning_rate", step=epoch)
    logger.log_scalar(train_result["stats"]["loss"], "train/loss", step=epoch)
    logger.log_scalar(
        train_result["stats"]["perplexity"], "train/perplexity", step=epoch
    )
    for result in validation_results:
        logger.log_scalar(
            result["stats"]["loss"], "{}/loss".format(result["name"]), epoch
        )
        logger.log_scalar(
            result["stats"]["perplexity"], "{}/perplexity".format(result["name"]), epoch
        )


def log_epoch_stats(
    logger: lavd.Logger,
    results: List[OrderedDict],
    metrics: List[OrderedDict],
    lr: Optional[float] = None,
    time_elapsed: Optional[float] = None,
    pad_prefix: bool = True,
):
    description = "{prefix}:".format(prefix=logger.prefix)
    if lr is not None:
        description += " Learning Rate = {lr:.8f}".format(lr=lr)
    if time_elapsed is not None:
        description += " (time elapsed {elapsed})".format(
            elapsed=time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
        )
    logger.println(description)
    header_names = ["Name"] + [metric["name"] for metric in metrics]
    line_values = []
    for result in results:
        values = [result["name"]]
        for metric in metrics:
            crit = result["stats"]
            for key in metric["key"].split("."):
                crit = crit.get(key)
                if crit is None:
                    break
            values.append(crit)
        line_values.append(values)
    logger.print_table(header_names, line_values, indent_level=1)
