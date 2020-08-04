import argparse
import csv
import json
import os
import tempfile

import torch
from halo import Halo
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from subword_nmt.learn_bpe import learn_bpe

seed = 1234
data_type = "leipzig"
min_prob = 0.99


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
        "--vocab", dest="vocab", action="store_true", help="Generate a vocabulary"
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
        "-t",
        "--type",
        dest="data_type",
        type=str,
        choices=["leipzig", "swiss-crawl"],
        default=data_type,
        help="Type of dataset to prepare [Default: {}]".format(data_type),
    )
    parser.add_argument(
        "-p",
        "--probability",
        dest="min_prob",
        type=float,
        default=min_prob,
        help=(
            "Minimum probability to keep a line "
            "(only applicable to the SwissCrawl dataset) "
            "[Default: {}]".format(min_prob)
        ),
    )
    return parser.parse_args()


def main():
    options = parse_args()
    torch.manual_seed(options.seed)
    basename = os.path.splitext(os.path.basename(options.input))[0]
    out_dir = options.out_dir or "data/{}/".format(basename)
    spinner = Halo(spinner="dots", placement="right")

    with open(options.input, "r") as fd:
        if options.data_type == "leipzig":
            reader = csv.reader(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            lines = [[line[1]] for line in reader]
        elif options.data_type == "swiss-crawl":
            reader = csv.reader(fd, delimiter=",")
            lines = []
            for i, line in enumerate(reader):
                # Skip the header
                if i == 0:
                    continue
                text = line[0]
                probability = float(line[2])
                if probability >= options.min_prob:
                    lines.append([text])
        else:
            raise Exception("Not a valid data type {}".format(options.data_type))

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

    if options.vocab:
        vocab_size = 32000
        spiece_out = os.path.join(out_dir, "spiece")
        spiece_args = (
            "--input={} "
            "--model_prefix={} "
            "--vocab_size={} "
            "--character_coverage=1.0"
        ).format(output_full, spiece_out, vocab_size)
        SentencePieceTrainer.Train(spiece_args)
        # Load the generated vocabulary
        with open("{}.vocab".format(spiece_out), "r") as fd:
            reader = csv.reader(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            vocab = [line[0] for line in reader]
        # Remove the special tokens <unk>, <s>, </s>
        vocab = vocab[3:]

        # Convert to BERT style
        bert_vocab = [
            v[1:] if v.startswith("▁") else "##{}".format(v) for v in vocab if v != "▁"
        ]
        # Add BERT's special tokens to the beginning
        bert_vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + bert_vocab
        # Fill up with unused tokens
        pad_size = vocab_size - len(bert_vocab)
        bert_vocab += ["unused{}".format(i) for i in range(pad_size)]
        with open(os.path.join(out_dir, "vocab.txt"), "w") as fd:
            writer = csv.writer(
                fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
            )
            writer.writerows([[b] for b in bert_vocab])

        # Convert to GPT-2 style
        # Unfortunately it's slow and tedious.
        spinner.start(text="Generating BPE vocabulary")
        gpt2_vocab = ["Ġ{}".format(v[1:]) if v.startswith("▁") else v for v in vocab]
        # Add the GPT-2 special token to the end
        gpt2_vocab.append("<|endoftext|>")
        with open(os.path.join(out_dir, "vocab.json"), "w") as fd:
            json.dump({v: i for i, v in enumerate(gpt2_vocab)}, fd, ensure_ascii=False)
        spiece_processor = SentencePieceProcessor()
        spiece_processor.Load("{}.model".format(spiece_out))
        # Encode the whole text
        encoded = [
            [" ".join(spiece_processor.EncodeAsPieces(line[0])).replace("▁", "Ġ")]
            for line in lines
        ]
        tmp_encoded_fd, tmp_encoded_path = tempfile.mkstemp()
        tmp_bpe_fd, tmp_bpe_path = tempfile.mkstemp()
        try:
            # Write the encoded text to a temporary file.
            with os.fdopen(tmp_encoded_fd, "w") as fd:
                writer = csv.writer(
                    fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
                )
                writer.writerows(encoded)
            learn_bpe(
                open(tmp_encoded_path, "r"),
                open(tmp_bpe_path, "w"),
                num_symbols=vocab_size,
            )
            with open(tmp_bpe_path, "r") as fd:
                reader = csv.reader(
                    fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
                )
                seen = set()
                merges = []
                for line in reader:
                    # Get rid of the </w> tokens
                    line = line[0].replace("</w>", "")
                    # Remove duplicates (due to </w> tokens)
                    if line not in seen:
                        seen.add(line)
                        merges.append([line])
            with open(os.path.join(out_dir, "merges.txt"), "w") as fd:
                writer = csv.writer(
                    fd, delimiter="\t", quoting=csv.QUOTE_NONE, quotechar=""
                )
                writer.writerows(merges)
        finally:
            os.remove(tmp_encoded_path)
            os.remove(tmp_bpe_path)
        spinner.stop()


if __name__ == "__main__":
    main()
