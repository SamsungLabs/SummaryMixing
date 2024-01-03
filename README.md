# SummaryMixing for SpeechBrain v0.5
*Halve your VRAM requirements and train 30% faster any speech model with SummaryMixing Conformers and Branchformers.*

## In brief
This repository implements SummaryMixing, a simpler, faster and much cheaper replacement to self-attention in Conformers and Branchformers for automatic speech recognition, keyword spotting and intent classification (see: the [publication](https://arxiv.org/abs/2307.07421) for further details). The code is fully compatible with the [SpeechBrain](https://speechbrain.github.io/) toolkit with version 0.5 -- copy and paste is all you need to start using SummaryMixing in your setup. If you wish to run with the latest version of SpeechBrain (v1.0+), please go to the main branch of this repository. SummaryMixing is the first alternative to MHSA able to beat it on speech tasks while reducing its complexity significantly (from quadratic to linear).

## A glance at SummaryMixing

SummaryMixing is a linear-time alternative to self-attention (SA) for speech processing models such as Transformers, Conformers or Branchformers. Instead of computing pair-wise scores between tokens (leading to quadratic-time complexity for SA), it summarises a whole utterance with mean over vectors for all time steps. SummaryMixing is based on the recent [findings](https://arxiv.org/pdf/2207.02971.pdf) demonstrating that self-attention could be useless for speech recognition as the attention weights of trained ASR systems are almost uniformly distributed accross the tokens composing a sequence. SummaryMixing also is a generalisation of the recent [HyperMixer](https://arxiv.org/abs/2203.03691) and [HyperConformer](https://arxiv.org/abs/2305.18281) to better and simpler mixing functions. In a SummaryMixing cell, that takes the same inputs and produces the same outputs than self-attention, contributions from each time step are first transformed and then averaged globally before being fed back to each time step. This is visible in Figure 1 in the [article](https://arxiv.org/abs/2307.07421). Therefore, the time-complexity is reduced to linear.

### A few results

A SummaryMixing-equipped Conformer outperforms a self-attention equivalent model on Librispeech test-clean (2.1% vs 2.3%) and test-other (5.1% vs 5.4%). This is done with a 30% training reduction as well as less than half of the memory budget (from 46GB to 21GB). Such gains are also visible with CommonVoice, AISHELL-1 and Tedlium2. This gain is also visible at decoding time as the real-time factor remains stable (does not increase) with the sentence length for a SummaryMixing Branchformer while the same model with self-attention would see its RTF following a quadratic increase. The SpeechBrain configuration files given in this repository are sufficient to reproduce these numbers.

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
This code is distributed under the CC-BY-NC 4.0 Licence. See the [Licence](#) for further details
