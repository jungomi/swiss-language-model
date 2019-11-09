import argparse
import csv
import os

import torch
from sentencepiece import SentencePieceTrainer

seed = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        type=str,
        help="Path to TSV file of the sentences",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="out_dir",
        type=str,
        help="Output directory [Default: data/<basename-of-input-file>/]",
    )
    parser.add_argument(
        "--split",
        dest="split",
        type=int,
        help=(
            "Percentage of split belonging to the training set, the rest belongs to "
            "the validation set. e.g. 80 means 80%% training and 20%% validation. "
            "Only if specified a train/validation split will be generated along side "
            "the full data."
        ),
    )
    parser.add_argument(
        "--no-vocab",
        dest="no_vocab",
        action="store_true",
        help="Do not generate a vocabulary",
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
    basename = os.path.splitext(os.path.basename(options.input))[0]
    out_dir = options.out_dir or "data/{}/".format(basename)

    with open(options.input, "r") as fd:
        reader = csv.reader(fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
        lines = [[line[1]] for line in reader]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_full = os.path.join(out_dir, "{}.tsv".format(basename))
    with open(output_full, "w") as fd:
        writer = csv.writer(fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar="")
        writer.writerows(lines)

    if options.split is not None:
        num_lines = len(lines)
        num_train = num_lines * options.split // 100
        num_validation = num_lines - num_train
        perms = torch.randperm(num_lines)
        train_indices, validation_indices = perms.split([num_train, num_validation])
        train_lines = [lines[i] for i in train_indices]
        validation_lines = [lines[i] for i in validation_indices]
        with open(os.path.join(out_dir, "{}-train.tsv".format(basename)), "w") as fd:
            writer = csv.writer(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            writer.writerows(train_lines)
        with open(
            os.path.join(out_dir, "{}-validation.tsv".format(basename)), "w"
        ) as fd:
            writer = csv.writer(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            writer.writerows(validation_lines)

    if not options.no_vocab:
        spiece_args = (
            "--input={} "
            "--model_prefix={} "
            "--vocab_size=32000 "
            "--character_coverage=1.0"
        ).format(output_full, os.path.join(out_dir, "spiece"))
        SentencePieceTrainer.Train(spiece_args)


if __name__ == "__main__":
    main()