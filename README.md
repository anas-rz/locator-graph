# LocatorGraph: Fault Localization in Sequence-Based Models using Graph Neural Networks

## Abstract

Deep learning models, particularly Sequence-based
models (SBMs) like Recurrent Neural Networks (RNNs), LSTM,
GRU, Transformer-based, and patch-based architectures play a
pivotal role in intelligent software systems. However, like any
software application, SBMs applications are susceptible to bugs.
Bug-fix patterns in SBMs differ from traditional techniques,
primarily due to their inherent black-box nature, data-driven
nature, and sequential dependencies. Moreover, current methods,
although tailored for generic deep neural network (DNN) structures, are time-consuming, require specialized expertise, and are
not directly applicable to the unique structure and requirements
of SBMs. To address these issues, we propose LocatorGraph, a
novel graph neural network-based tool to identify the root causes
of faults in SBMs.
We convert SBM code to trace graphs which are then fed
to the proposed framework LocatorGraph, a data-driven tool,
that employs a graph neural network model to trace graphs
for SBMs, enabling it to effectively detect and localize bugs
in real-world SBMs. Beyond mere identification, LocatorGraph
locates faulty nodes in the generated tracegraphs after which
the bug-causing faulty features are located for repair. Through
rigorous evaluation of 160 diverse models including generated
buggy models, LocatorGraph outperforms existing methods in
fault localization, showing robustness in identifying potential
problems with an AUC of 79.39% and F1 score of 80.08%. On
Node Detection, LG gives an accuracy of 75.84% and an F1-score
of 76.26%. In the final phase, LG outperforms all baselines to
locate bug-causing features in the nodes with an accuracy of
74.32% and F1 score 73.21%. We also provide details of the
improvement observed in the models after fixing bugs using
_LocatorGraph_ based approach.

# Setup

Experiments have been performed in notebooks at Google Colab and the High-Performance Cluster at Oakland University.

Please checkout [this](./locatorgraph) to explore experimentation.



# Training
