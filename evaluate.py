import argparse
import multiprocessing
import os
import time
from collections import OrderedDict
from typing import Dict

import lavd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)

from checkpoint import load_checkpoint, log_epoch_stats, metrics
from dataset import TextDataset, mask_tokens

batch_size = 1
num_workers = multiprocessing.cpu_count()
num_gpus = torch.cuda.device_count()
seed = 1234


def evaluate(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    logger: lavd.Logger,
    masked_lm: bool = True,
    name: str = "",
) -> Dict:
    # Disables autograd during validation mode
    torch.set_grad_enabled(False)
    model.eval()

    sampler = (
        data_loader.sampler  # type: ignore
        if isinstance(data_loader.sampler, DistributedSampler)  # type: ignore
        else None
    )

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

        output = (
            model(inputs, masked_lm_labels=labels)
            if masked_lm
            else model(inputs, labels=labels)
        )
        loss = output[0]
        losses.append(loss.item())

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        dest="datasets",
        nargs="+",
        metavar="[NAME=]PATH",
        required=True,
        type=str,
        help=(
            "List of text files to evaluate. "
            "If no name is specified it uses the name of the text file."
        ),
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint",
        required=True,
        nargs="+",
        type=str,
        help="Paths to the checkpoints to be evaluated",
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
        "--no-cuda",
        dest="no_cuda",
        action="store_true",
        help="Do not use CUDA even if it's available",
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        default=seed,
        type=int,
        help="Seed for random initialisation [Default: {}]".format(seed),
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
        options.num_workers = options.num_workers // options.num_gpus
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
    for cp in options.checkpoint:
        checkpoint = load_checkpoint(os.path.join(cp, "stats.pth"))
        name = "evaluate/{}".format(cp)
        logger = lavd.Logger(name, disabled=gpu_id != 0)

        spinner = logger.spinner("Initialising")
        spinner.start()

        # All but the primary GPU wait here, so that only the primary process loads the
        # pre-trained model and the rest uses the cached version.
        if distributed and gpu_id != 0:
            torch.distributed.barrier()

        model_kind = checkpoint["model"].get("kind")
        use_special = True
        masked_lm = True
        add_space = False
        if model_kind == "bert" or model_kind == "bert-scratch":
            config = BertConfig.from_pretrained(cp)
            model = BertForMaskedLM.from_pretrained(cp, config=config)
            tokeniser = BertTokenizer.from_pretrained(cp)
        elif model_kind == "gpt2" or model_kind == "gpt2-scratch":
            config = GPT2Config.from_pretrained(cp)
            model = GPT2LMHeadModel.from_pretrained(cp, config=config)
            tokeniser = GPT2Tokenizer.from_pretrained(cp)
            masked_lm = False
            use_special = False
            add_space = True
        else:
            raise Exception("No model available for {}".format(model_kind))
        model = model.to(device)

        # Primary process has loaded the model and the other can now load the cached
        # version.
        if distributed and gpu_id == 0:
            torch.distributed.barrier()

        data_loaders = []
        for data_file in options.datasets:
            data = data_file.split("=", 1)
            if len(data) > 1:
                # Remove whitespace around the name
                name = data[0].strip()
                # Expand the ~ to the full path as it won't be done automatically since
                # it's not at the beginning of the word.
                file_path = os.path.expanduser(data[1])
            else:
                name = None
                file_path = data[0]
            dataset = TextDataset(
                file_path,
                tokeniser,
                name=name,
                use_special=use_special,
                add_space=add_space,
            )
            sampler = (
                DistributedSampler(
                    dataset, num_replicas=options.num_gpus, rank=gpu_id, shuffle=False
                )
                if distributed
                else None
            )
            data_loader = DataLoader(
                dataset,
                batch_size=options.batch_size,
                shuffle=False,
                num_workers=options.num_workers,
                sampler=sampler,
                pin_memory=True,
            )
            data_loaders.append(data_loader)

        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[gpu_id], find_unused_parameters=True
            )

        # Wait for all processes to load eveything before starting training.
        # Not strictly necessary, since they will wait once the actual model is run, but
        # this makes it nicer to show the spinner until all of them are ready.
        if distributed:
            torch.distributed.barrier()
        spinner.stop()

        start_time = time.time()
        logger.set_prefix("Evaluation - {}".format(cp))
        results = []
        for data_loader in data_loaders:
            data_name = data_loader.dataset.name
            logger.start(data_name)
            result = evaluate(
                data_loader,
                model,
                device=device,
                name=data_name,
                logger=logger,
                masked_lm=masked_lm,
            )
            result["name"] = data_name
            results.append(result)
            logger.end(data_name)

        time_difference = time.time() - start_time
        evaluation_results = [
            OrderedDict(
                name=result["name"],
                stats=OrderedDict(loss=result["loss"], perplexity=result["perplexity"]),
            )
            for result in results
        ]
        log_epoch_stats(
            logger, evaluation_results, metrics, time_elapsed=time_difference
        )


if __name__ == "__main__":
    main()
