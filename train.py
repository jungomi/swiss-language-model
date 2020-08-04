import argparse
import os
import time
from collections import OrderedDict
from typing import Dict, Optional

import lavd
import torch
import torch.cuda.amp as amp
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
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)

from checkpoint import (
    default_checkpoint,
    load_checkpoint,
    log_epoch_stats,
    log_experiment,
    log_results,
    log_top_checkpoints,
    metrics,
    save_checkpoint,
)
from dataset import TextDataset, mask_tokens

batch_size = 1
num_workers = mp.cpu_count()
num_gpus = torch.cuda.device_count()
num_epochs = 100

lr = 5e-5
adam_eps = 1e-8
lr_warmup = 0
weight_decay = 0.0
seed = 1234
default_model = "bert"


def run_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    optimiser: optim.Optimizer,  # type: ignore
    device: torch.device,
    logger: lavd.Logger,
    epoch: int,
    train: bool = True,
    amp_scaler: Optional[amp.GradScaler] = None,
    masked_lm: bool = True,
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
    pbar = logger.progress_bar(
        name, total=len(data_loader.dataset), leave=False, dynamic_ncols=True
    )
    tokeniser = data_loader.dataset.tokeniser  # type: ignore
    for d in data_loader:
        d = d.to(device)
        inputs, labels = mask_tokens(d, tokeniser) if masked_lm else (d, d)
        # The last batch may not be a full batch
        curr_batch_size = inputs.size(0)

        # Automatically run it in mixed precision (FP16) if a scaler is given
        with amp.autocast(enabled=amp_scaler is not None):
            output = (
                model(inputs, masked_lm_labels=labels)
                if masked_lm
                else model(inputs, labels=labels)
            )
        loss = output[0]
        losses.append(loss.item())
        if torch.isnan(loss) or torch.isinf(loss):
            breakpoint()
        if train:
            optimiser.zero_grad()
            if amp_scaler is None:
                loss.backward()
                # Clip gradients to avoid exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimiser.step()
            else:
                amp_scaler.scale(loss).backward()
                amp_scaler.unscale_(optimiser)
                # Clip gradients to avoid exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                amp_scaler.step(optimiser)
                amp_scaler.update()

        pbar.update(
            curr_batch_size
            if sampler is None
            else curr_batch_size * sampler.num_replicas  # type: ignore
        )

    pbar.close()

    loss = torch.mean(torch.tensor(losses, device=device))
    # Gather the loss onto the primary process to have accurate metrics.
    if sampler is not None:
        gathered_losses = [
            torch.zeros_like(loss) for _ in range(sampler.num_replicas)  # type: ignore
        ]
        dist.all_gather(gathered_losses, loss)
        loss = torch.mean(torch.tensor(gathered_losses))
    perplexity = torch.exp(loss)
    return OrderedDict(loss=loss.item(), perplexity=perplexity.item())


def train(
    logger: lavd.Logger,
    model: nn.Module,
    optimiser: optim.Optimizer,  # type: ignore
    train_data_loader: DataLoader,
    validation_data_loaders: DataLoader,
    lr_scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    checkpoint: Dict,
    num_epochs: int = num_epochs,
    model_kind: str = default_model,
    amp_scaler: Optional[amp.GradScaler] = None,
    masked_lm: bool = True,
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
            else OrderedDict(
                start=start_epoch, stats=OrderedDict(loss=[], perplexity=[])
            )
        )
        validation_results_dict[val_name] = val_result

    # All validations that are no longer used, will be stored in outdated_validation
    # just to have them available.
    outdated_validations.append(
        OrderedDict(
            {k: v for k, v in validation_cp.items() if k not in validation_results_dict}
        )
    )

    tokeniser = train_data_loader.dataset.tokeniser  # type: ignore
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
            amp_scaler=amp_scaler,
            masked_lm=masked_lm,
        )
        train_stats["stats"]["loss"].append(train_result["loss"])
        train_stats["stats"]["perplexity"].append(train_result["perplexity"])
        epoch_lr = lr_scheduler.get_last_lr()[0]  # type: ignore
        train_stats["lr"].append(epoch_lr)
        lr_scheduler.step()
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
                amp_scaler=amp_scaler,
                masked_lm=masked_lm,
            )
            validation_results.append(
                OrderedDict(name=val_name, stats=validation_result)
            )
            validation_results_dict[val_name]["stats"]["loss"].append(
                validation_result["loss"]
            )
            validation_results_dict[val_name]["stats"]["perplexity"].append(
                validation_result["perplexity"]
            )
            logger.end(val_text)

        with logger.spinner("Checkpoint", placement="right"):
            # Multi-gpu models wrap the original model. To make the checkpoint
            # compatible with the original model, the state dict of .module is saved.
            model_unwrapped = (
                model.module if isinstance(model, DistributedDataParallel) else model
            )
            save_checkpoint(
                logger,
                model_unwrapped,
                tokeniser,
                stats=OrderedDict(
                    epoch=actual_epoch,
                    train=train_stats,
                    validation=validation_results_dict,
                    outdated_validation=outdated_validations,
                    model=OrderedDict(kind=model_kind),
                ),
                step=actual_epoch,
            )

        with logger.spinner("Logging Data", placement="right"):
            log_results(
                logger,
                actual_epoch,
                OrderedDict(lr=epoch_lr, stats=train_result),
                validation_results,
                model_unwrapped,
            )

        with logger.spinner("Best Checkpoints", placement="right"):
            val_stats = OrderedDict(
                {
                    val_name: {
                        "name": val_name,
                        "start": val_result["start"],
                        "stats": val_result["stats"],
                    }
                    for val_name, val_result in validation_results_dict.items()
                }
            )
            log_top_checkpoints(logger, val_stats, metrics)

        time_difference = time.time() - start_time
        epoch_results = [
            OrderedDict(name="Train", stats=train_result)
        ] + validation_results
        log_epoch_stats(
            logger, epoch_results, metrics, lr=epoch_lr, time_elapsed=time_difference
        )
        logger.end(epoch_text, prefix=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-text",
        dest="train_text",
        required=True,
        type=str,
        help="Path to text file for training",
    )
    parser.add_argument(
        "--validation-text",
        dest="validation_text",
        nargs="+",
        metavar="[NAME=]PATH",
        default=[],
        type=str,
        help=(
            "List of text files validation. "
            "If no name is specified it uses the name of the text file."
        ),
    )
    parser.add_argument(
        "--pre-trained",
        dest="pre_trained",
        type=str,
        help="Which pre-trained model to use",
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
        "--name", dest="name", type=str, help="Name of the experiment",
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
        choices=["bert", "bert-scratch", "gpt2", "gpt2-scratch", "gpt2-german"],
        help="Which kind of model to use [Default: {}]".format(default_model),
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        help="Enable mixed precision training (FP16)",
    )
    parser.add_argument(
        "--vocab",
        dest="vocab",
        type=str,
        help="Directory with the vocabulary to use (only for models from scratch)",
    )
    return parser


def main():
    options = build_parser().parse_args()
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
        options.actual_batch_size = options.batch_size // options.num_gpus
        options.actual_num_workers = (
            options.num_workers + options.num_gpus - 1
        ) // options.num_gpus
        mp.spawn(run, nprocs=options.num_gpus, args=(options, True))
    else:
        options.actual_batch_size = options.batch_size
        options.actual_num_workers = options.num_workers
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
    torch.manual_seed(options.seed)
    use_cuda = torch.cuda.is_available() and not options.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")
    logger = lavd.Logger(options.name, disabled=gpu_id != 0)
    # Parser needs to be rebuilt, since it can't be serialised and it is needed to even
    # detect the number of GPUs, but here it's only used to log it.
    parser = build_parser() if gpu_id == 0 else None

    spinner = logger.spinner("Initialising")
    spinner.start()

    checkpoint = (
        default_checkpoint
        if options.checkpoint is None
        else load_checkpoint(os.path.join(options.checkpoint, "stats.pt"))
    )
    # Either use the checkpoint directory as the configuration or use one of the
    # available pre-trained models.
    pre_trained = options.checkpoint or options.pre_trained

    # All but the primary GPU wait here, so that only the primary process loads the
    # pre-trained model and the rest uses the cached version.
    if distributed and gpu_id != 0:
        torch.distributed.barrier()

    model_kind = checkpoint["model"].get("kind") or options.model_kind
    use_special = True
    masked_lm = True
    if model_kind == "bert":
        if pre_trained is None:
            pre_trained = "bert-base-german-cased"
        config = BertConfig.from_pretrained(pre_trained)
        model = BertForMaskedLM.from_pretrained(pre_trained, config=config)
        tokeniser = BertTokenizer.from_pretrained(pre_trained)
    elif model_kind == "bert-scratch":
        # The pre_trained here is only for the configuartion (num layers etc.)
        # But the weights are not loaded
        if pre_trained is None:
            pre_trained = "bert-base-german-cased"
        # Use either the provided vocabulary or the pre_trained one.
        vocab = options.vocab or pre_trained
        tokeniser = BertTokenizer.from_pretrained(vocab)
        config = BertConfig.from_pretrained(pre_trained)
        config.vocab_size = tokeniser.vocab_size
        model = BertForMaskedLM(config)
    elif model_kind == "gpt2":
        if pre_trained is None:
            pre_trained = "gpt2"
        config = GPT2Config.from_pretrained(pre_trained)
        model = GPT2LMHeadModel.from_pretrained(pre_trained, config=config)
        tokeniser = GPT2Tokenizer.from_pretrained(pre_trained)
        masked_lm = False
        use_special = False
    elif model_kind == "gpt2-german":
        assert pre_trained is not None, "--pre-trained must be given for gpt2-german"
        config = GPT2Config.from_pretrained(pre_trained)
        model = GPT2LMHeadModel.from_pretrained(pre_trained, config=config)
        # Using the XLNetTokenizer because the pre-trained German GPT-2 model uses
        # SentencePiece and that's easiest way to use it.
        # That also means that the automatic tokenisation cannot be done, because XLNet
        # uses different placing of the special tokens.
        tokeniser = XLNetTokenizer.from_pretrained(
            pre_trained,
            keep_accents=True,
            unk_token="<unk>",
            # start and end of sequence use the same token
            bos_token="<endoftext>",
            eos_token="<endoftext>",
        )
        masked_lm = False
        use_special = False
    elif model_kind == "gpt2-scratch":
        # The pre_trained here is only for the configuartion (num layers etc.)
        # But the weights are not loaded
        if pre_trained is None:
            pre_trained = "gpt2"
        # Use either the provided vocabulary or the pre_trained one.
        vocab = options.vocab or pre_trained
        tokeniser = GPT2Tokenizer.from_pretrained(vocab)
        config = GPT2Config.from_pretrained(pre_trained)
        config.vocab_size = tokeniser.vocab_size
        model = GPT2LMHeadModel(config)
        masked_lm = False
        use_special = False
    else:
        raise Exception("No model available for {}".format(model_kind))
    model = model.to(device)

    # Primary process has loaded the model and the other can now load the cached
    # version.
    if distributed and gpu_id == 0:
        torch.distributed.barrier()

    train_dataset = TextDataset(
        options.train_text,
        tokeniser,
        use_special=use_special,
        manual_special=model_kind == "gpt2-german",
    )
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
        num_workers=options.actual_num_workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    validation_data_loaders = []
    for val_file in options.validation_text:
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
        validation_dataset = TextDataset(
            file_path,
            tokeniser,
            name=name,
            use_special=use_special,
            manual_special=model_kind == "gpt2-german",
        )
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
            num_workers=options.actual_num_workers,
            sampler=validation_sampler,
            pin_memory=True,
        )
        validation_data_loaders.append(validation_data_loader)

    initial_lr = options.lr
    # Only restore the learning rate if resuming from a checkpoint and not manually
    # resetting the learning rate.
    if len(checkpoint["train"]["lr"]) > 0 and not options.reset_lr:
        initial_lr = checkpoint["train"]["lr"][-1]

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
    lr_scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=options.lr_warmup,
        num_training_steps=options.num_epochs,
    )

    amp_scaler = amp.GradScaler() if use_cuda and options.fp16 else None

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
    log_experiment(logger, experiment)

    logger.log_command(parser, options)

    # Wait for all processes to load eveything before starting training.
    # Not strictly necessary, since they will wait once the actual model is run, but
    # this makes it nicer to show the spinner until all of them are ready.
    if distributed:
        torch.distributed.barrier()
    spinner.stop()

    if options.checkpoint is not None:
        resume_text = "Resuming from - Epoch {epoch}".format(epoch=checkpoint["epoch"])
        logger.set_prefix(resume_text)
        epoch_results = [
            OrderedDict(
                name="Train",
                stats=OrderedDict(
                    loss=checkpoint["train"]["stats"]["loss"][-1],
                    perplexity=checkpoint["train"]["stats"]["perplexity"][-1],
                ),
            )
        ] + [
            OrderedDict(
                name=val_name,
                stats=OrderedDict(
                    loss=val_result["stats"]["loss"][-1],
                    perplexity=val_result["stats"]["perplexity"][-1],
                ),
            )
            for val_name, val_result in checkpoint["validation"].items()
        ]
        log_epoch_stats(logger, epoch_results, metrics)

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
        amp_scaler=amp_scaler,
        masked_lm=masked_lm,
    )


if __name__ == "__main__":
    main()
