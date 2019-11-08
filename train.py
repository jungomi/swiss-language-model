import argparse
import multiprocessing
import os
import time
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    WarmupLinearSchedule,
)

from checkpoint import Logger, Noop, default_checkpoint, load_checkpoint, metrics
from dataset import TextDataset, mask_tokens


batch_size = 1
num_workers = multiprocessing.cpu_count()
num_gpus = torch.cuda.device_count()
num_epochs = 100
lr = 5e-5
adam_eps = 1e-8
lr_warmup = 0
weight_decay = 0.0
seed = 1234
default_model = "bert"
default_name = "default"
pre_trained = "bert-base-german-cased"


def run_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    optimiser: optim.Optimizer,  # type: ignore
    device: torch.device,
    logger: Logger,
    epoch: int,
    train: bool = True,
    name: str = "",
) -> Dict:
    # Disables autograd during validation mode
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()

    sampler = (
        data_loader.sampler  # type: ignore
        if isinstance(data_loader.sampler, DistributedSampler)  # type: ignore
        else None
    )
    if sampler is not None:
        sampler.set_epoch(epoch)

    losses = []
    logger.progress_start(name, total=len(data_loader.dataset))
    tokeniser = data_loader.dataset.tokeniser  # type: ignore
    for d in data_loader:
        inputs, labels = mask_tokens(d.to(device), tokeniser)
        # The last batch may not be a full batch
        curr_batch_size = inputs.size(0)

        output = model(inputs, masked_lm_labels=labels)
        loss = output[0]
        losses.append(loss.item())
        if torch.isnan(loss) or torch.isinf(loss):
            breakpoint()
        if train:
            optimiser.zero_grad()
            loss.backward()
            # Clip gradients to avoid exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        logger.progress_step(
            curr_batch_size
            if sampler is None
            else curr_batch_size * sampler.num_replicas  # type: ignore
        )

    logger.progress_end()

    loss = torch.mean(torch.tensor(losses))
    perplexity = torch.exp(loss)
    return OrderedDict(loss=loss.item(), perplexity=perplexity.item())


def train(
    logger: Logger,
    model: nn.Module,
    optimiser: optim.Optimizer,  # type: ignore
    train_data_loader: DataLoader,
    validation_data_loaders: DataLoader,
    lr_scheduler: WarmupLinearSchedule,
    device: torch.device,
    checkpoint: Dict,
    num_epochs: int = num_epochs,
    model_kind: str = default_model,
    sample_image: Optional[str] = None,
):
    start_epoch = checkpoint["epoch"]
    train_stats = checkpoint["train"]
    validation_cp = checkpoint["validation"]
    outdated_validations = checkpoint["outdated_validation"]

    validation_results_dict: Dict[str, Dict] = OrderedDict()
    for val_data_loader in validation_data_loaders:
        val_name = val_data_loader.dataset.name
        val_result = (
            validation_cp[val_name]
            if val_name in validation_cp
            else OrderedDict(start=start_epoch, loss=[], perplexity=[])
        )
        validation_results_dict[val_name] = val_result

    # All validations that are no longer used, will be stored in outdated_validation
    # just to have them available.
    outdated_validations.append(
        OrderedDict(
            {k: v for k, v in validation_cp.items() if k not in validation_results_dict}
        )
    )

    for epoch in range(num_epochs):
        actual_epoch = start_epoch + epoch + 1
        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=epoch + 1,
            end=num_epochs,
            epoch=actual_epoch,
            pad=len(str(num_epochs)),
        )
        logger.set_prefix(epoch_text)
        logger.start(epoch_text, prefix=False)
        start_time = time.time()

        logger.start("Train")
        train_result = run_epoch(
            train_data_loader,
            model,
            optimiser,
            device=device,
            epoch=epoch,
            train=True,
            name="Train",
            logger=logger,
        )
        train_stats["loss"].append(train_result["loss"])
        train_stats["perplexity"].append(train_result["perplexity"])
        train_result["name"] = "Train"
        logger.end("Train")

        validation_results = []
        for val_data_loader in validation_data_loaders:
            val_name = val_data_loader.dataset.name
            val_text = "Validation: {}".format(val_name)
            logger.start(val_text)
            validation_result = run_epoch(
                val_data_loader,
                model,
                optimiser,
                device=device,
                epoch=epoch,
                train=False,
                name=val_text,
                logger=logger,
            )
            validation_result["name"] = val_name
            validation_results.append(validation_result)
            val_stats = validation_results_dict[val_name]
            val_stats["loss"].append(validation_result["loss"])
            val_stats["perplexity"].append(validation_result["perplexity"])
            logger.end(val_text)

        epoch_lr = lr_scheduler.get_lr()[0]
        lr_scheduler.step()
        train_result["lr"] = epoch_lr
        train_stats["lr"].append(epoch_lr)

        logger.start("Checkpoint", spinner=True)
        logger.save_checkpoint(
            OrderedDict(
                epoch=actual_epoch,
                train=train_stats,
                validation=validation_results_dict,
                outdated_validation=outdated_validations,
                # Multi-gpu models wrap the original model. To make the checkpoint
                # compatible with the original model, the state dict of .module is
                # saved.
                model=OrderedDict(
                    kind=model_kind,
                    state=(
                        model.module.state_dict()
                        if isinstance(model, DistributedDataParallel)
                        else model.state_dict()
                    ),
                ),
            )
        )
        logger.end("Checkpoint", spinner=True)

        logger.start("Tensorboard", spinner=True)
        logger.write_tensorboard(
            actual_epoch,
            train_result,
            validation_results,
            (model.module if isinstance(model, DistributedDataParallel) else model),
        )
        logger.end("Tensorboard", spinner=True)

        logger.start("Log Best Checkpoint", spinner=True)
        logger.log_top_checkpoints(validation_results_dict, metrics)
        logger.end("Log Best Checkpoint", spinner=True)

        time_difference = time.time() - start_time
        epoch_results = [
            OrderedDict(
                name="Train",
                loss=train_result["loss"],
                perplexity=train_result["perplexity"],
            )
        ] + [
            OrderedDict(
                name=val_result["name"],
                loss=val_result["loss"],
                perplexity=val_result["perplexity"],
            )
            for val_result in validation_results
        ]
        logger.log_epoch_stats(
            epoch_results, metrics, lr=epoch_lr, time_elapsed=time_difference
        )
        logger.end(epoch_text, prefix=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file",
        dest="train_file",
        required=True,
        type=str,
        help="Path to text file for training",
    )
    parser.add_argument(
        "--validation-file",
        dest="validation_file",
        required=True,
        nargs="+",
        metavar="[NAME=]PATH",
        type=str,
        help=(
            "List of text files validation. "
            "If no name is specified it uses the name of the text file."
        ),
    )
    parser.add_argument(
        "--pre-trained",
        dest="pre_trained",
        default=pre_trained,
        type=str,
        help="Which pre-trained model to use [Default: {}]".format(pre_trained),
    )
    parser.add_argument(
        "-n",
        "--num-epochs",
        dest="num_epochs",
        default=num_epochs,
        type=int,
        help="Number of epochs to train [Default: {}]".format(num_epochs),
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        default=batch_size,
        type=int,
        help="Size of data batches [Default: {}]".format(batch_size),
    )
    parser.add_argument(
        "-w",
        "--workers",
        dest="num_workers",
        default=num_workers,
        type=int,
        help="Number of workers for loading the data [Default: {}]".format(num_workers),
    )
    parser.add_argument(
        "-g",
        "--gpus",
        dest="num_gpus",
        default=num_gpus,
        type=int,
        help="Number of GPUs to use [Default: {}]".format(num_gpus),
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        dest="lr",
        default=lr,
        type=float,
        help="Learning rate to use [Default: {}]".format(lr),
    )
    parser.add_argument(
        "--reset-lr",
        dest="reset_lr",
        action="store_true",
        help=(
            "Reset the learning rate to the specified value if a checkpoint is given "
            "instead of continuing from the checkpoint's learning rate"
        ),
    )
    parser.add_argument(
        "--lr-warmup",
        dest="lr_warmup",
        default=lr_warmup,
        type=int,
        help="Number of warmup steps for the Learning rate [Default: {}]".format(
            lr_warmup
        ),
    )
    parser.add_argument(
        "--adam-eps",
        dest="adam_eps",
        default=adam_eps,
        type=float,
        help="Epsilon for the Adam optimiser [Default: {}]".format(adam_eps),
    )
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        default=weight_decay,
        type=float,
        help="Weight decay of the optimiser [Default: {}]".format(weight_decay),
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        help="Path to the checkpoint to be loaded to resume training",
    )
    parser.add_argument(
        "--no-cuda",
        dest="no_cuda",
        action="store_true",
        help="Do not use CUDA even if it's available",
    )
    parser.add_argument(
        "--name",
        dest="name",
        default=default_name,
        type=str,
        help="Name of the experiment",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        default=seed,
        type=int,
        help="Seed for random initialisation [Default: {}]".format(seed),
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model_kind",
        default=default_model,
        choices=["bert"],
        help="Which kind of model to use [Default: {}]".format(default_model),
    )

    return parser.parse_args()


def main():
    options = parse_args()
    torch.manual_seed(options.seed)
    use_cuda = torch.cuda.is_available() and not options.no_cuda
    if use_cuda:
        # Somehow this fixes an unknown error on Windows.
        torch.cuda.current_device()

    # Get rid of the annoying warnings about TensorFlow not being compiled with
    # certain CPU instructions.
    # TensorFlow is not even used, but because transformers uses it besides PyTorch
    # there are constant warnings being spammed.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if use_cuda and options.num_gpus > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        # Manullay adjust the batch size and workers to split amongst the processes.
        options.batch_size = options.batch_size // options.num_gpus
        options.num_workers = (
            options.num_workers + options.num_gpus - 1
        ) // options.num_gpus
        mp.spawn(run, nprocs=options.num_gpus, args=(options, True))
    else:
        run(0, options)


def run(gpu_id, options, distributed=False):
    if distributed:
        dist.init_process_group(
            backend="nccl",
            rank=gpu_id,
            world_size=options.num_gpus,
            init_method="env://",
        )
        torch.cuda.set_device(gpu_id)
    use_cuda = torch.cuda.is_available() and not options.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    logger = Logger(options.name) if gpu_id == 0 else Noop()

    logger.start("Initialising", spinner=True, prefix=False)

    tokeniser = BertTokenizer.from_pretrained(options.pre_trained)

    train_dataset = TextDataset(options.train_file, tokeniser)
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=options.num_gpus, rank=gpu_id)
        if distributed
        else None
    )
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        # Only shuffle when not using a sampler
        shuffle=train_sampler is None,
        num_workers=options.num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    validation_data_loaders = []
    for val_file in options.validation_file:
        vals = val_file.split("=", 1)
        if len(vals) > 1:
            # Remove whitespace around the name
            name = vals[0].strip()
            # Expand the ~ to the full path as it won't be done automatically since it's
            # not at the beginning of the word.
            file_path = os.path.expanduser(vals[1])
        else:
            name = None
            file_path = vals[0]
        validation_dataset = TextDataset(file_path, tokeniser, name=name)
        validation_sampler = (
            DistributedSampler(
                validation_dataset, num_replicas=options.num_gpus, rank=gpu_id
            )
            if distributed
            else None
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=options.batch_size,
            # Only shuffle when not using a sampler
            shuffle=validation_sampler is None,
            num_workers=options.num_workers,
            sampler=validation_sampler,
            pin_memory=True,
        )
        validation_data_loaders.append(validation_data_loader)

    checkpoint = None
    model_kind = options.model_kind

    initial_lr = options.lr
    checkpoint = (
        default_checkpoint
        if options.checkpoint is None
        else load_checkpoint(options.checkpoint)
    )
    # Only restore the learning rate if resuming from a checkpoint and not manually
    # resetting the learning rate.
    if len(checkpoint["train"]["lr"]) > 0 and not options.reset_lr:
        initial_lr = checkpoint["train"]["lr"][-1]

    # All but the primary GPU wait here, so that only the primary process loads the
    # pre-trained model and the rest uses the cached version.
    if distributed and gpu_id != 0:
        torch.distributed.barrier()

    if model_kind == "bert":
        config = BertConfig.from_pretrained(options.pre_trained)
        model = BertForMaskedLM.from_pretrained(options.pre_trained, config=config)
    else:
        raise Exception("No model available for {}".format(model_kind))
    model = model.to(device)

    # Primary process has loaded the model and the other can now load the cached
    # version.
    if distributed and gpu_id == 0:
        torch.distributed.barrier()

    if options.checkpoint is not None:
        resume_text = "Resuming from - Epoch {epoch}".format(epoch=checkpoint["epoch"])
        logger.set_prefix(resume_text)
        epoch_results = [
            OrderedDict(
                name="Train",
                loss=checkpoint["train"]["loss"][-1],
                perplexity=checkpoint["train"]["perplexity"][-1],
            )
        ] + [
            OrderedDict(
                name=val_name,
                loss=val_result["loss"][-1],
                perplexity=val_result["perplexity"][-1],
            )
            for val_name, val_result in checkpoint["validation"].items()
        ]
        logger.log_epoch_stats(epoch_results, metrics)

    no_decay = ["bias", "LayerNorm.weight"]
    optimiser_grouped_parameters = [
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            "weight_decay": options.weight_decay,
        },
        {
            "params": [
                param
                for name, param in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimiser = AdamW(optimiser_grouped_parameters, lr=initial_lr, eps=options.adam_eps)
    lr_scheduler = WarmupLinearSchedule(
        optimiser,
        warmup_steps=options.lr_warmup,
        t_total=len(train_dataset) * options.num_epochs,
    )

    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[gpu_id], find_unused_parameters=True
        )

    validation_details = [
        OrderedDict(
            name=data_loader.dataset.name,
            path=data_loader.dataset.path,
            size=len(data_loader.dataset),
        )
        for data_loader in validation_data_loaders
    ]
    experiment = OrderedDict(
        model_kind=model_kind,
        train=OrderedDict(path=train_dataset.path, size=len(train_dataset)),
        validation=validation_details,
        options=options,
    )
    logger.log_experiment(experiment)

    # Wait for all processes to load eveything before starting training.
    # Not strictly necessary, since they will wait once the actual model is run, but
    # this makes it nicer to show the spinner until all of them are ready.
    if distributed:
        torch.distributed.barrier()
    logger.end("Initialising", spinner=True, prefix=False)

    train(
        logger,
        model,
        optimiser,
        train_data_loader,
        validation_data_loaders,
        lr_scheduler=lr_scheduler,
        device=device,
        num_epochs=options.num_epochs,
        checkpoint=checkpoint,
        model_kind=model_kind,
    )


if __name__ == "__main__":
    main()
