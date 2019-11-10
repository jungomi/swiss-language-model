# Swiss Language Model

A language model for Swiss German.

Using [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding][arxiv-bert]
pre-trained on [cased German text by Deepset.ai][bert-german], which included:
German Wikipedia dump (6GB of raw txt files), the OpenLegalData dump (2.4 GB)
and news articles (3.6 GB)

The model is then fine tuned on the Swiss German data of the
[Leipzig Corpora Collection][leipzig-corpora].

## Requirements

- Python 3
- [PyTorch][pytorch]

All dependencies can be installed with PIP.

```sh
pip install --user -r requirements.txt
```

If you'd like to use a different installation method or another CUDA version
with PyTorch follow the instructions on
[PyTorch - Getting Started][pytorch-started].

### Apex - Mixed Precision Training (Optional)

Modern GPUs contain Tensor Cores (starting from V100 and RTX series) which
enable mixed precision calculation, using optimised fp16 operations while still
keeping the fp32 weights and therefore precision.

*Other GPUs without Tensor Cores do not benefit from using mixed precision
since they only do fp32 operations and you may find it even becoming slower.*

[Apex][apex] needs to be installed to enable mixed precision training:

```sh
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --user --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

Then the `-O`/`--opt-level` can be set to use the different optimisation levels.

- O0: No optimisations (fp32)
- O1: Mixed precision (Recommended)
- O2: Almost fp16 but still mixed precision
- O3: Full fp16 (Almost guaranteed to lose accuracy)


## Usage

### Data

Data for training essentially raw text files, but since the Leipzig corpus uses
a TSV style, that has been kept, but instead of the second column containing the
sentences (first one in Leipzig corpus is the index), it is now the first one.
This means you can add more columns after the first one, if you have a dataset
that needs additional labels (e.g. for sentiment of the sentence) or just any
additional information that will be ignored during training.

The Leipzig corpus can be converted with `prepare_data.py`:

```sh
python prepare_data.py -i data/leipzig.tsv -o leipzig-converted --split 80
```

`-o`/`--output` is the output directory for all the generated data. If not
given, it will be generated to `data/<basename-of-input-file>`, which would be
`data/leipzig/` in this example.
`--split` optionally generates a training validation split (80/20 in this case)
additionally to the full data.
You can also generate a vocabulary, SentencePiece and WordPiece (BERT's style)
from the input by supplying the `--vocab` flag.

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
[bert-german]: https://deepset.ai/german-bert
[leipzig-corpora]: https://wortschatz.uni-leipzig.de/en/download/
[pytorch]: https://pytorch.org/
[pytorch-started]: https://pytorch.org/get-started/locally/
