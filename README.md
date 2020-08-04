# Swiss Language Model

![Python CI Badge](https://github.com/jungomi/swiss-language-model/workflows/Python/badge.svg)

A language model for Swiss German based on
[Huggingface/Transformers][huggingface-transformers].

Using [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding][arxiv-bert]
pre-trained on [cased German text by Deepset.ai][bert-german], which included:
German Wikipedia dump (6GB of raw txt files), the OpenLegalData dump (2.4 GB)
and news articles (3.6 GB)

The model is then fine tuned on the Swiss German data of the
[Leipzig Corpora Collection][leipzig-corpora] and
[SwissCrawl][swiss-crawl-corpus].

Alternatively, a GPT-2 model can also be trained, but there is no German
pre-trained model available for that.

## Requirements

- Python 3
- [PyTorch][pytorch]
- [Huggingface/Transformers][huggingface-transformers]

All dependencies can be installed with pip.

```sh
pip install --user -r requirements.txt
```

On *Windows* the PyTorch packages may not be available on PyPi, hence you need
to point to the official PyTorch registry:

```sh
pip install --user -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

If you'd like to use a different installation method or another CUDA version
with PyTorch follow the instructions on
[PyTorch - Getting Started][pytorch-started].

## Models

| Model           | Configuration | Datasets used for training | Download                      |
|-----------------|---------------|----------------------------|-------------------------------|
| `bert-swiss-lm` | `bert-base`   | Leipzig and SwissCrawl     | [Google Drive][bert-swiss-lm] |

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
You can also generate a vocabulary, SentencePiece, WordPiece (BERT's style) and
Byte Pair Encoding (BPE) from the input by supplying the `--vocab` flag.

Similarly, the SwissCrawl data can be prepared by setting the `-t`/`--type`
option to `swiss-crawl`. For this preparation a minimum probability of 0.99 is
used by default and can be changed with `-p`/`--p`.

```sh
python prepare_data.py -i data/swiss-crawl.csv -o swiss-crawl-converted --split 80 -t swiss-crawl
```

### Training

Training is done with the `train.py` script:

```sh
python train.py --name some-name -c log/some-name/checkpoints/0022/ --train-text /path/to/text.tsv --validation-text /path/to/text.tsv --fp16
```

The `--name` option is used to give it a name, otherwise the checkpoints are
just numbered without any given name and `-c` is to resume from the given
checkpoint, if not specified it starts fresh.

Modern GPUs contain Tensor Cores (starting from V100 and RTX series) which
enable mixed precision calculation, using optimised fp16 operations while still
keeping the fp32 weights and therefore precision.

It can be enabled by setting the `--fp16` flag.

*Other GPUs without Tensor Cores do not benefit from using mixed precision
since they only do fp32 operations and you may find it even becoming slower.*

Different models can be selected with the `-m`/`--model` option, which are
either `bert` or `gpt2`, to fine tune a pre-trained model, which can be changed
with the `--pre-trained` option by specifying one model available at
[Huggingface Transformers - Pretrained Models][huggingface-pre-trained] or by
specifyiing a path to the pre-trained model.
There's also the possibility to train either of the model from scratch by
choosing `bert-scratch` or `gpt-scratch` for the `--model`. The configuration
used can still be changed with the `--pre-trained` option, but the pre-trained
weights will not be loaded, just the configuration. Addtionally, the vocabulary
can be changed with `--vocab` (path to the directory of the generated
vocabularies) if another vocbaulary instead of the pre-trained one should be
used.

For all options see `python train.py --help`.

#### Logs

During the training various types of logs are created with [Lavd][lavd] and
everything can be found in `log/` and is grouped by the experiment name.

- Summary
- Checkpoints
- Top 5 Checkpoints
- TensorBoard
- Event logs

To visualise the logged data run:

```sh
lavd log/
```

[arxiv-bert]: https://arxiv.org/abs/1810.04805
[bert-german]: https://deepset.ai/german-bert
[bert-swiss-lm]: https://drive.google.com/open?id=1FBIIMO9C1Os-Er7DpL2G2DuUbsjWP2ts
[huggingface-transformers]: https://github.com/huggingface/transformers
[huggingface-pre-trained]: https://huggingface.co/transformers/pretrained_models.html
[lavd]: https://github.com/jungomi/lavd
[leipzig-corpora]: https://wortschatz.uni-leipzig.de/en/download/
[pytorch]: https://pytorch.org/
[pytorch-started]: https://pytorch.org/get-started/locally/
[swiss-crawl-corpus]: https://icosys.ch/swisscrawl
