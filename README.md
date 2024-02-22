# Cross-Linguistic Syntactic Difference in mBERT

This repository contains code for the EMNLP 2022 Paper [*Cross-Linguistic Syntactic Difference in Multilingual BERT: How Good is It and How Does It Affect Transfer*](https://aclanthology.org/2022.emnlp-main.552).

We have integrated the code from this repository into our [Calf](https://github.com/ningyuxu/calf) framework.

## Data

### Universal Dependencies

The Universal Dependencies (UD) [version 2.8](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3687) Treebank is used for our experiments, and we follow the split of training, development and test set in UD.



## Experiments

### Multilingual BERT

We use the pretrained [bert-base-multilingual-cased model](https://huggingface.co/bert-base-multilingual-cased) for all our experiments.

### Grammatical relation probe

We train the linear classifier via [stochastic gradient descent](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) to classify the grammatical relations between a head-dependent word pair. 


### Measure of Syntactic Difference in mBERT

We use the [public source code](https://github.com/microsoft/otdd) of Alvarez-Melis and Fusi (2020) to compute the syntactic difference in mBERT. The p-Wasserstein distance (p = 2) is computed based on Sinkhorn algorithm (Cuturi, 2013) and the entropy regularization strength is set to 1e-1.

### Dependency parsing

We follow the setup of Wu and Dredze (2019), which replaces the LSTM encoder in Dozat and Manning (2017) with mBERT. For each language, we train the model with ten epochs and validate it at the end of each epoch. We choose the model performing the best (i.e., achieving the highest LAS) on the development set. We use the Adam optimizer with β1 = 0.9, β2 = 0.99, eps = 1e-8, and a learning rate of 5e-5. The batch size is 16 and the max sequence length is set to 128.

### Gradient Boosting Regressor

We use a [gradient boosting regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) with 100 estimators and each has a maximum depth of 3. We use the squared error for regression with the default learning rate of 1e-1.


## Reference

Alvarez-Melis, D., & Fusi, N. (2020). Geometric Dataset Distances via Optimal Transport. Advances in Neural Information Processing Systems, 33, 21428–21439. https://proceedings.neurips.cc/paper/2020/hash/f52a7b2610fb4d3f74b4106fb80b233d-Abstract.html

Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. Advances in Neural Information Processing Systems, 26. https://papers.nips.cc/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html

Wu, S., & Dredze, M. (2019). Beto, Bentz, Becas: The Surprising Cross-Lingual Effectiveness of BERT. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 833–844. https://doi.org/10.18653/v1/D19-1077
