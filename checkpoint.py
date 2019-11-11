import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from halo import Halo
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

default_checkpoint = {
    "epoch": 0,
    "train": {"lr": [], "loss": [], "perplexity": []},
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


def noop(*args, **kwargs):
    return None


class Noop(object):
    """
    A No-Op class that always returns the noop function.
    Useful to stub out the Logger.
    """

    def __getattr__(self, _):
        return noop


class Logger(object):
    """Logger for anything related to the training"""

    def __init__(
        self, name: str, dir: str = "./log", delimiter: str = "\t", train: bool = True
    ):
        super(Logger, self).__init__()
        self.name = name
        self.delimiter = delimiter
        self.created_timestamp = datetime.now()
        self.log_dir = os.path.join(dir, name)
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        self.git_hash = (
            subprocess.check_output(["git", "-C", repo_path, "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        self.events_file = open(
            os.path.join(self.log_dir, "events.log"), "w", buffering=1
        )
        self.output_file = open(
            os.path.join(self.log_dir, "output.log"), "w", buffering=1
        )
        self.events_time: Dict[str, float] = {}
        self.spinner = Halo(spinner="dots", placement="right")
        self.prefix = ""
        # Only create checkpoints and tensorboard when training.
        if train:
            self.tensorboard = SummaryWriter(os.path.join(self.log_dir, "tensorboard"))
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

    def __del__(self):
        self.events_file.close()
        self.output_file.close()

    def set_prefix(self, prefix: str):
        self.prefix = prefix

    def start_time(self) -> str:
        return self.created_timestamp.strftime("%Y-%m-%d %H:%M:%S")

    def start(self, name: str, spinner: bool = False, prefix: bool = True):
        self.events_time[name] = time.time()
        msg = "{} - {}".format(self.prefix, name) if prefix else name
        self.log("START", msg)
        if spinner:
            self.spinner.start(text=msg)

    def end(self, name: str, spinner: bool = False, prefix: bool = True):
        start_time = self.events_time.pop(name, None)
        end_time = time.time()
        msg = "{} - {}".format(self.prefix, name) if prefix else name
        if start_time is not None:
            time_difference = end_time - start_time
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time_difference))
            msg = "{} - Duration: {}".format(msg, elapsed_time)
        self.log("END", msg)
        if spinner:
            self.spinner.stop()

    def println(self, msg: str, *args, **kwargs):
        formatted_msg = msg.format(*args, **kwargs)
        self.output_file.write(formatted_msg)
        self.output_file.write("\n")
        self.log("STDOUT", formatted_msg)
        print(formatted_msg)

    def log(self, tag: str, msg: str):
        now = datetime.now()
        self.events_file.write(str(now))
        self.events_file.write(self.delimiter)
        self.events_file.write(tag)
        self.events_file.write(self.delimiter)
        self.events_file.write(msg)
        self.events_file.write("\n")

    def log_top_checkpoints(
        self, results: Dict[str, Dict], criterion: List[OrderedDict], k: int = 5
    ):
        with open(os.path.join(self.log_dir, "best.md"), "w") as fd:
            fd.write("# {} - Best Checkpoints\n".format(self.name))
            for name, result in results.items():
                fd.write("\n")
                fd.write("## {}\n".format(name))
                for metric in criterion:
                    fd.write("\n")
                    fd.write("### {}\n\n".format(metric["name"]))
                    crit = result
                    for key in metric["key"].split("."):
                        crit = crit[key]
                    values = torch.tensor(crit)
                    descending = metric["order"] == "max"
                    sorted_values, sorted_indices = torch.sort(
                        values, descending=descending
                    )
                    for i, (value, index) in enumerate(
                        zip(sorted_values.tolist(), sorted_indices.tolist())
                    ):
                        if i >= k:
                            break
                        fd.write(
                            (
                                "{i}. log/{name}/checkpoints/{num:0>4}/ "
                                "- {value:.5f}\n"
                            ).format(
                                i=i + 1,
                                name=self.name,
                                num=result["start"] + index + 1,
                                value=value,
                            )
                        )

    def log_experiment(self, experiment: Dict):
        diff = subprocess.check_output(["git", "-C", repo_path, "diff", "HEAD"]).decode(
            "utf-8"
        )
        diff_file = os.path.join(self.log_dir, "changes.patch")
        if len(diff) > 0:
            with open(diff_file, "w") as fd:
                fd.write(diff)
        with open(os.path.join(self.log_dir, "summary.md"), "w") as fd:
            fd.write("# {}\n\n".format(self.name))
            fd.write("- **Start**: {}\n".format(self.start_time()))
            fd.write("- **Git Commit**: {}\n".format(self.git_hash))
            fd.write("- **Model**: {}\n".format(experiment["model_kind"]))
            fd.write("- **Train Dataset**\n")
            fd.write("    - *Size*: {}\n".format(experiment["train"]["size"]))
            fd.write("    - *Path*: {}\n".format(experiment["train"]["path"]))
            fd.write("- **Validation Datasets**:\n")
            for val in experiment["validation"]:
                fd.write("    - **{name}**\n".format(name=val["name"]))
                fd.write("        - *Size*: {}\n".format(val["size"]))
                fd.write("        - *Path*: {}\n".format(val["path"]))
            fd.write("\n")

            fd.write("## Command\n\n")
            fd.write("```sh\n")
            command = " ".join(["python", *sys.argv])
            fd.write(command)
            fd.write("\n")
            fd.write("```\n\n")

            fd.write("## Restore Working Tree\n\n")
            fd.write(
                "The working tree at the time of the experiment can be restored with:"
            )
            fd.write("\n\n")
            fd.write("```sh\n")
            fd.write("git checkout {}\n".format(self.git_hash))
            if len(diff) > 0:
                fd.write("git apply {}\n".format(diff_file))
            fd.write("```\n\n")

            fd.write("## Options\n\n")
            fd.write("```\n")
            options_dict = vars(experiment["options"])
            for k, v in options_dict.items():
                fd.write("{} = {}\n".format(k, v))
            fd.write("```\n\n")

            if len(diff) > 0:
                fd.write("\n")
                fd.write("## Git Changes\n\n")
                fd.write(
                    "There were uncommitted changes when running the experiment\n\n"
                )
                fd.write("```diff\n")
                fd.write(diff)
                fd.write("```\n")

    def save_checkpoint(
        self, model: PreTrainedModel, tokeniser: PreTrainedTokenizer, checkpoint: Dict
    ):
        # Padded to 4 digits because of lexical sorting of numbers.
        # e.g. 0009
        dir_num = "{num:0>4}".format(num=checkpoint["epoch"])
        out_dir = os.path.join(self.checkpoint_dir, dir_num)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model.save_pretrained(out_dir)
        tokeniser.save_pretrained(out_dir)
        torch.save(checkpoint, os.path.join(out_dir, "stats.pth"))

    def write_tensorboard(
        self,
        epoch: int,
        train_result: Dict,
        validation_results: List[Dict],
        model: nn.Module,
    ):
        self.tensorboard.add_scalar("learning_rate", train_result["lr"], epoch)
        self.tensorboard.add_scalar("train/loss", train_result["loss"], epoch)
        self.tensorboard.add_scalar(
            "train/perplexity", train_result["perplexity"], epoch
        )
        for result in validation_results:
            self.tensorboard.add_scalar(
                "{}/loss".format(result["name"]), result["loss"], epoch
            )
            self.tensorboard.add_scalar(
                "{}/perplexity".format(result["name"]), result["perplexity"], epoch
            )
        for name, param in model.named_parameters():
            self.tensorboard.add_histogram(
                "{}".format(name), param.detach().cpu().numpy(), epoch
            )
            if param.grad is not None:
                self.tensorboard.add_histogram(
                    "{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch
                )

    def log_epoch_stats(
        self,
        results: List[OrderedDict],
        metrics: List[OrderedDict],
        lr: Optional[float] = None,
        time_elapsed: Optional[float] = None,
    ):
        description = "{prefix}:".format(prefix=self.prefix)
        if lr is not None:
            description += " Learning Rate = {lr:.5f}".format(lr=lr)
        if time_elapsed is not None:
            description += " (time elapsed {elapsed})".format(
                elapsed=time.strftime("%H:%M:%S", time.gmtime(time_elapsed))
            )
        self.println(description)
        prefix_pad = " " * len(self.prefix)
        header_names = ["Name"] + [metric["name"] for metric in metrics]
        header_lengths = [len(name) for name in header_names]
        line_values = []
        # The lengths for all fields in all lines, including the header. Used to find
        # the maximum width per field to generate a nice table.
        field_lengths = [header_lengths]
        for result in results:
            values = [result["name"]]
            field_lens = [len(result["name"])]
            for metric in metrics:
                crit = result
                for key in metric["key"].split("."):
                    crit = crit[key]
                val = "{:.5f}".format(crit)
                values.append(val)
                field_lens.append(len(val))
            line_values.append(values)
            field_lengths.append(field_lens)
        max_field_lengths = torch.max(torch.tensor(field_lengths), dim=0)[0].tolist()
        header_names = pad_fields(header_names, max_field_lengths)
        header = "| {names} |".format(names=" | ".join(header_names))
        self.println("{pad}{line}", pad=prefix_pad, line=header)
        delimiter = re.sub("[^|]", "-", header)
        self.println("{pad}{line}", pad=prefix_pad, line=delimiter)
        for values in line_values:
            line = " | ".join(pad_fields(values, max_field_lengths))
            self.println("{pad}| {line} |", pad=prefix_pad, line=line)

    def progress_start(self, name: str, total: int, prefix: bool = True):
        self.pbar = tqdm(
            desc="{} ({})".format(self.prefix, name) if prefix else name,
            total=total,
            dynamic_ncols=True,
            leave=False,
        )

    def progress_end(self):
        self.pbar.close()

    def progress_step(self, size: int = 1):
        self.pbar.update(size)


def pad_fields(fields: List[str], lengths: List[int], value=" ") -> List[str]:
    return [f + value * (length - len(f)) for f, length in zip(fields, lengths)]
