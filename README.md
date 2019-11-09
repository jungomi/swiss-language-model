# Swiss Language Model

A language model for Swiss German.

Using [BERT][arxiv-bert]

## Requirements

- Python 3
- [PyTorch][pytorch]

All dependencies can be installed with PIP.

```sh
pip install -r requirements.txt
```

If you'd like to use a different installation method or another CUDA version
with PyTorch follow the instructions on
[PyTorch - Getting Started][pytorch-started].

### Apex - Mixed Precision Training (Optional)

For mixed precision training (f16) [Apex][apex] needs to be installed:

```sh
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --user --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Then the `-O`/`--opt-level` can be set to use the different optimisation levels.

- O0: No optimisations (f32)
- O1: Mixed precision (Recommended)
- O2: Almost f16 but still mixed precision
- O3: Full f16 (Almost guaranteed to lose accuracy)


## Usage

### Training

Training is done with the `train.py` script:

```sh
python train.py --name some-name -c log/some-name/checkpoints/0022/ --train-text /path/to/text.tsv --validation-text /path/to/text.tsv
```

The `--name` option is used to give it a name, otherwise the checkpoints are
just numbered without any given name and `-c` is to resume from the given
checkpoint, if not specified it starts fresh.

For all options see `python train.py --help`.

#### Logs

During the training various types of logs are created and everything can be
found in `log/` and is grouped by the experiment name.

- Summary
- Checkpoints
- Top 5 Checkpoints
- TensorBoard
- Event logs

Even though they are grouped by the experiment, TensorBoard automatically finds
all of them, therefore it can be run with:

```sh
tensorboard --logdir log
```

[apex]: https://github.com/nvidia/apex
[arxiv-bert]: https://arxiv.org/abs/1810.04805
[pytorch]: https://pytorch.org/
[pytorch-started]: https://pytorch.org/get-started/locally/
