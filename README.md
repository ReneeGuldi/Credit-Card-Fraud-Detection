# Credit Card Fraud Detection

A PyTorch Neural Network that detects fraudulent credit card transactions in a highly imbalanced dataset, where fraud makes up only 0.17% fo all transactions.

## Overview

Only 492 of 284,807 credit card transactions in this dataset are fraudulent (0.17%), which makes accuracy a misleading metric. A model that labels everything "not fraud" would still score 99.83%. 
This project compares a logistic regression baseline against a feed-forward neural network (30 --> 64 --> 32 --> 1, ReLU, Adam, BCE loss) built in PyTorch. The Neural Network is evaluated with a full confusion matrix, precision, recall, and F1 score rather than accuracy alone.

## Results:

On a held-out test set of 56,962 transactions, the neural network caught 74 of 98 fraud cases (75.5% recall) with only 10 false alarms (88.1% precision, F1 = 0.81)

| Metric | Neural Network |
| --- | --- |
| Precision | 88.1% |
| Recall | 75.5% |
| F1 Score | 0.81 |
| Fraud Caught | 74/98 |
| False Alarms | 10 of 56,864 legitimate transactions |

The logistic regression baseline reached 99.1% accuracy, compared to 99.94% for the neural network.

## Tech Stack

Python · PyTorch · pandas · NumPy · Matplotlib

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (284,807 transactions, 492 fraud cases). The file is too large to include in this repository (~150 MB).

## How to Run

The notebook was built in Google Colab. To run it:

1. Open `Project1_Renee-Guldi.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter.
2. Run all cells. The notebook downloads the dataset automatically using 'gdown'.
     - Alternatively download `creditcard.csv` from Kaggle, and place it in the same folder as the notebook.
3. If running locally, install dependencies first: `pip install torch pandas numpy matplotlib gdown`

## What I learned

- **Machine learning can detect fraud automatically.** Even without any special handling for the imbalance, the neural network correctly flagged 74 fraudulent transactions while raising only 10 false alarms across nearly 57,000 legitimate ones. Reviewing that many transactions by hand would be slow and error-prone.
- **Neural networks are well-suited to complex patterns.** The hiden layers and ReLU activations let the network learn non-linear relationships between the 30 features that a single linear layer like logistic regression can't capture.
- **Class imbalance is the real challenge.** With fraud making up only 0.17% of the data, both models reached over 99.9% accuracy, which looks impressive, but actually said very little. The confusion matrix showed the real story: the model still missed 24 of 98 fraud cases. This taught me to choose metrics based on the problem rather than defaulting to accuracy.

## Future Improvements

- Oversample fraud cases or weight the loss function so the model pays more attention to the rare cases.
- Lower the decision threshold from 0.5 to attempt to catch more fraud, trading a few more false alarms for higher recall.
- Evaluate the logistic regression baseline with the same confusion-matrix for a direct comparison. 

## Files

- `Project1_Renee-Guldi.ipynb`: data loading, preprocessing, model training, and evaluation
- `Guldi-Renee_Project1-Presentation.pptx`: project presentation slides


## Acknowledgments

Data-splitting and scaling helper functions were provided as part of CSCI-460 course materials.
