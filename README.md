# Cross-Domain-Transfer-of-Hyperspectral-Foundation-Models
This repository is the official implementation of the paper:

**Cross-Domain Transfer of Hyperspectral Foundation Models** <br/>
Nick Theisen, Peer Neubert </br>
*International Conference on Pattern Recognition (ICPR), 2026*

## Abstract

Hyperspectral imaging (HSI) semantic segmentation typically relies on in-domain training, but limited data availability often restricts model performance in real-world applications. Current approaches to leverage foundation models in proximal sensing use cross-modality techniques, bridging RGB and HSI to exploit vision foundation models. However, these methods either discard spectral information or introduce architectural complexity. We propose cross-domain transfer as an alternative, reusing HSI foundation models – originally trained in remote sensing – for proximal sensing applications. By eliminating the need to bridge modality gaps, our approach preserves spectral information while maintaining a simple architecture. Using the HS3-Bench benchmark, we systematically evaluate and compare conventional in-domain, in-modality training, cross-modality transfer and cross-domain transfer strategies. Our results demonstrate that cross-domain transfer achieves large performance improvements over in-domain, in-modality training,reduces the performance gap to cross-modality approaches and maintains strong performance in limited data settings. Thus, this work advances more effective HSI semantic segmentation in diverse applications.

## Acknowledgement
This repository reuses code from the HS3-Bench repository https://github.com/nickstheisen/hyperseg

This work was partially funded by Wehrtechnische Dienststelle 41 (WTD), Koblenz, Germany.
