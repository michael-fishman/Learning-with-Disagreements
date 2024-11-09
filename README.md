# Project Overview

This repository houses the "Learning with Disagreements" project, which innovatively approaches annotator disagreements not as noise, but as valuable signals that can inform and enhance labeling strategies. By recognizing disagreements as data-rich signals, we develop and evaluate various methods to harness these insights effectively. Our approach includes a novel ensemble method that synthesizes different labeling strategies to enhance model accuracy and robustness. We rigorously test each method's effectiveness individually and in combination across diverse datasets.

## Project Philosophy

Traditional approaches often view annotator disagreements as mere noise that complicates data labeling. Contrary to this view, our project treats these disagreements as crucial indicators that reveal subtleties and complexities within the data which can significantly inform the training process. This shift in perspective allows us to explore more sophisticated and nuanced methods of data annotation.

## Project Components

- **Ensemble Method**: Our ensemble method combines labels from Bayesian, Crowd Truth, and Majority Voting approaches, weighted according to their reliability and the context of the data. This methodology is implemented in `Ensemble_Method.py`.
- **Individual Method Evaluation**: `Deberta_Pipeline.py` serves as a pipeline for independently assessing the impact of each labeling method on model performance, allowing us to understand the strengths and weaknesses of each approach in isolation.
  
## Files and Folders

- **Learning_with_Disagreements_Crowd_Truth.ipynb**: This Jupyter notebook implements the Crowd Truth method for label generation.
- **bayesian_method.py**: Script for generating labels using the Bayesian method.
- **Deberta_Pipeline.py**: Evaluates each labeling method separately to ascertain their impact on model training and testing.
- **Ensemble_Method.py**: Contains the implementation of our novel ensemble labeling method.
- **Final-Data-sets/**: Directory holding all datasets utilized in the project, complete with labels from each method.

## Data Sources

- **GoEmotions dataset**: [Explore on Hugging Face](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- **Measuring Hate Speech dataset**: [Available on Hugging Face](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech)
- **Social Bias Frames**: [Details here](https://maartensap.com/social-bias-frames/)

## Key References

- **The Problem of Human Label Variation**: [Read the paper](https://arxiv.org/abs/2211.02570)
- **Crowd Truth Methodology**: [Learn more here](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2564)
- **Bayesian Method for Labeling**: [Detailed information](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=c6e8d5e9ca4e58c413857b7fbd3a11054f2262cb)

## Contribution

Contributions are welcome to refine our methods or expand the scope of datasets. For specific instructions on how to contribute, please refer to the documentation within each script.
