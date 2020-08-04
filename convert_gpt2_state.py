import argparse
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import GPT2Model

regex_blocks = re.compile("^blocks")
regex_bias = re.compile("\\.b$")
regex_weight = re.compile("\\.g$")


def rename_key(key: str) -> str:
    key = regex_blocks.sub("h", key)
    key = regex_bias.sub(".bias", key)
    key = regex_weight.sub(".weight", key)
    return key


def reshape_weight(key: str, weight: torch.Tensor) -> torch.Tensor:
    if (
        "attn.c_attn.weight" in key
        or "mlp.c_fc.weight" in key
        or "mlp.c_proj.weight" in key
    ):
        # Those weights have swapped dimensions
        return weight.transpose(0, 1)
    return weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the pre-trained German GPT2 to work "
            "with HuggingFace's transformers"
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        type=str,
        help="Path to pre-trained model file",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        required=True,
        type=str,
        help="Output directory to save the converted model",
    )
    return parser.parse_args()


def main():
    options = parse_args()
    input_path = Path(options.input)
    if input_path.is_dir():
        input_path = input_path / "model.pt"
    checkpoint = torch.load(input_path, map_location="cpu")
    converted_state = {
        rename_key(key): reshape_weight(key, value)
        for key, value in checkpoint["state_dict"].items()
    }
    gpt2 = GPT2Model.from_pretrained("gpt2-medium")
    # The vocab is smaller than the actual gpt2 one, therefore it is padded with zeros
    # to match it. The zeros will be unused.
    gpt2_vocab_size = gpt2.wte.weight.size(0)
    vocab_size = converted_state["wte.weight"].size(0)
    pad_size = gpt2_vocab_size - vocab_size
    converted_state["wte.weight"] = F.pad(
        converted_state["wte.weight"], [0, 0, 0, pad_size], mode="constant", value=0.0
    )

    # There are some weights that are not in the pre-trained model, which will be
    # trained in the down stream task. As long as no key from the pre-trained model did
    # not match one of the actual keys, it should be fine.
    incompatible_keys = gpt2.load_state_dict(converted_state, strict=False)
    assert (
        len(incompatible_keys.unexpected_keys) == 0
    ), "Unexpected keys in the model: {}".format(incompatible_keys.unexpected_keys)

    gpt2.save_pretrained(options.output)


if __name__ == "__main__":
    main()
