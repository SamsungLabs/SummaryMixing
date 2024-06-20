# SummaryMixing wav2vec 2.0
We equip wav2vec 2.0 (w2v2) with SummaryMixing, our linear-time alternative to the quadratic cost self-attention. Compared to self-attention based w2v2, SummaryMixing based w2v2 greatly reduces the cost for self-supervised pre-training and gives better or the same level performance on downstream tasks. 

## In brief
This repository implements SummaryMixing w2v2. The code is fully compatible with the [SpeechBrain](https://speechbrain.github.io/) copy and paste is all you need to start using SummaryMixing in your setup.

## A glance at SummaryMixing

SummaryMixing is a linear-time alternative to self-attention (SA) for speech processing models such as Transformers, Conformers or Branchformers. Instead of computing pair-wise scores between tokens (leading to quadratic-time complexity for SA), it summarises a whole utterance with mean over vectors for all time steps. SummaryMixing is based on the recent [findings](https://arxiv.org/pdf/2207.02971.pdf) demonstrating that self-attention could be useless for speech recognition as the attention weights of trained ASR systems are almost uniformly distributed accross the tokens composing a sequence. SummaryMixing also is a generalisation of the recent [HyperMixer](https://arxiv.org/abs/2203.03691) and [HyperConformer](https://arxiv.org/abs/2305.18281) to better and simpler mixing functions. In a SummaryMixing cell, that takes the same inputs and produces the same outputs than self-attention, contributions from each time step are first transformed and then averaged globally before being fed back to each time step. This is visible in Figure 1 in the [article](https://arxiv.org/abs/2307.07421). Therefore, the time-complexity is reduced to linear.

In this branch, we use SummaryMixing for self-supervised learning by equipping w2v2 with SummaryMixing. For a detailed description, please refer to this [article]()

### A few results

In the experiment of the [article](), SummaryMixing-equipped w2v2 reduces the pre-training time and memory budget by 18% and 23%, respectively, with better or equivalent results for the downstream automatic speech recognition, intent classification, emotion recognition, and automatic speaker verification. The following Table gives the results of SummaryMixing-based and attention-based SSL models on CommonVoice Welsh ASR and SLURP intent classification. For the results of other downstream tasks please refer to the [article](). The SpeechBrain configuration files in this repository can reproduce these numbers. 


| Context Encoder | Size     | Pre-trained on | Welsh 15.8 WER | SLURP Intent Classification Acc. | 
|------------------|----------------------|--------------------|---------------------|
| Self-attention   | 166M   | LibriLight 4.3k h                | 50.8                 | 78.1                 |
| SummaryMixing    | 155M     | LibriLight 4.3k h                | 48.3                 | 80.5                | 
|------------------|----------------------|--------------------|---------------------|---------------------|
| w2v2 base        | 95M     | LibriSpeech 960 h       | 54.5                | 77.7           |
| w2v2 large       | 317M     | LibriLight 60k h                | 45.4                 | 79.0        |


## Citation

Please cite SummaryMixing as follows:
```bibtex
@misc{summarymixing,
  title={{SummaryMixing}: A Linear-Complexity Alternative to Self-Attention for Speech Recognition and Understanding},
  author={Titouan Parcollet and Rogier van Dalen and and Shucong Zhang and Sourav Bhattacharya},
  year={2023},
  eprint={2307.07421},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2307.07421}
}
```

## Licence
This code is distributed under the CC-BY-NC 4.0 Licence. See the [Licence](https://github.com/SamsungLabs/SummaryMixing/blob/main/LICENCE.md) for further details
