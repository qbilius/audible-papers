# Citation removal tool

## Overview

This tools attempts to remove in-line citations from academic papers. For instance, if you have a sentence like

```
It has been reported (Smith et al., 2024) that ...
```

this tool should return

```
It has been reported that ...
```

But why?

Because sometimes I like to *listen* to academic papers rather than read them, and these references break the flow.


## Installation

```
pip install torch torchmetrics lightning numpy datasets tiktoken tqdm rich
```


## Usage

### Training (optional)

Training a tiny attention model to learn to classify what is a citation and what is not.

```
python train.py -c config.yaml fit
```

### Inference

Removes whatever is determined to be a citation and save as a text file.


### Listen

1. Save the resulting text file at a cloud storage, e.g., Dropbox.
2. Open it up using [@Voice Aloud Reader](https://www.hyperionics.com/atvoice/) on Android or some other text-to-speech synthesizer on other platforms.


## License

MIT