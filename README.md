# Retrieval is Accurate Generation

This is the repository for our paper [Retrieval is Accurate Generation](https://arxiv.org/abs/2402.17532).

Standard language models generate text by selecting tokens from a fixed, finite, and standalone vocabulary. We introduce a novel method that selects context-aware phrases from a collection of supporting documents. One of the most significant challenges for this paradigm shift is determining the training oracles, because a string of text can be segmented in various ways and each segment can be retrieved from numerous possible documents. To address this, we propose to initialize the training oracles using linguistic heuristics and, more importantly, bootstrap the oracles through iterative self-reinforcement. Extensive experiments show that our model not only outperforms standard language models on a variety of knowledge-intensive tasks but also demonstrates improved generation quality in open-ended text generation. For instance, compared to the standard language model counterpart, our model raises the accuracy from 23.47% to 36.27% on OpenbookQA, and improves the MAUVE score from 42.61% to 81.58% in open-ended text generation. Remarkably, our model also achieves the best performance and the lowest latency among several retrieval-augmented baselines. In conclusion, we assert that retrieval is more accurate generation and hope that our work will encourage further research on this new paradigm shift.

Please see [our paper](https://arxiv.org/pdf/2402.17532) for more details.



## 🛠️ Installation
To set up the environment, run the following command:
```
pip3 install -r requirements.txt
```


## 📈 Usage
Here's a quick guide on how to use this repository:


### Training:
Assuming that the corpus is given, we first need to extract good phrase candidates from it.
```
```
Train the query encoder:
```
bash scripts/train_pipeline.sh 0,1,2,3,4,5,6,7
```
### Testing:
First, use the trained model to encode all phrase candidates to obtain a phrase index.
```
bash scripts/encode_phrase.sh 0,1,2,3,4,5,6,7
```
For open-ended text generation:
```
```
For QA tasks:
```
```

## 📬 Contact

For any questions, feel free to open an issue or contact us directly at [bwcao@link.cuhk.edu.hk].
