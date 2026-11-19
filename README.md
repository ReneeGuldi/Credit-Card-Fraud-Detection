# Credit Card Fraud Detection

A PyTorch neural network that detects fraudulent credit card transactions in a highly imbalanced dataset, where fraud makes up only 0.17% of all transactions. This README walks through how I approached the problem, not just where it ended up.

## The Problem

Only 492 of 284,807 transactions in this dataset are fraudulent. Before writing any model code, I counted the classes and realized what that meant: a model that labels *everything* "not fraud" would still score 99.83% accuracy while catching zero fraud.

That changed the question I was trying to answer. Instead of asking "how accurate is the model?", I asked **"how many fraud cases does it catch, and how many honest customers does it bother along the way?"**

## My Approach

### 1. Explore and prepare the data
- Loaded the data with pandas and confirmed its shape: 30 input features and 1 label (`Class`).
- Split it 80/20 into training (227,845) and test (56,962) sets, then standardized the features so no single feature (like `Amount`) dominated training.
- Converted everything into PyTorch tensors, using the GPU when one was available.

### 2. Start with a simple baseline
Before building anything complex, I trained a **logistic regression** model, which is a single linear layer with a sigmoid. That gave me a reference point. It reached 99.91% accuracy, which looked great until I remembered the 99.83% "do-nothing" score. The baseline showed me that accuracy alone couldn't tell the models apart.

### 3. Build the neural network
Next I built a feed-forward network to see whether hidden layers could pick up non-linear patterns that a linear model misses:

- **Architecture:** 30 → 64 → 32 → 1, ReLU activations, sigmoid output
- **Training:** Adam optimizer, binary cross-entropy loss, mini-batches of 256, 20 epochs

### 4. Evaluate with the right metrics
Because accuracy was misleading, I evaluated the network with a **full confusion matrix**, plus precision, recall, and F1. That's where the real story showed up.

## Results

On the held-out test set of 56,962 transactions:

| Metric | Neural Network |
| --- | --- |
| Precision | 88.1% |
| Recall | 75.5% |
| F1 Score | 0.81 |
| Fraud caught | 74 of 98 |
| Fraud missed | 24 |
| False alarms | 10 of 56,864 legitimate transactions |

The network's accuracy was 99.94%, but the number I care about more is the 24 missed fraud cases. That points directly to what I'd work on next.

## What I Learned

- **Pick metrics based on the problem, not by default.** Both models scored above 99.9% accuracy, which said almost nothing. The confusion matrix was what showed the model still missed about 1 in 4 fraud cases.
- **Baselines keep you honest.** Training logistic regression first gave me something to compare against and made the accuracy trap obvious early.
- **Hidden layers matter for complex patterns.** ReLU activations let the network learn non-linear relationships between the 30 features that a single linear layer can't capture.
- **Machine learning can take on tedious review work.** Flagging 74 fraud cases with only 10 false alarms across nearly 57,000 legitimate transactions would be slow and error-prone to do by hand.

## What I'd Try Next

Each of these targets the 24 missed cases:

- **Weight the loss or oversample fraud** so the model pays more attention to the rare class (I didn't handle the imbalance at all in this version).
- **Lower the decision threshold** from 0.5 to trade a few more false alarms for higher recall.
- **Evaluate the logistic regression baseline with the same confusion matrix** for a true side-by-side comparison.

## Tech Stack

Python · PyTorch · pandas · NumPy · Matplotlib

## Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (284,807 transactions by European cardholders over two days in September 2013, 492 fraud cases). The file is too large to include in this repository (~150 MB).

## How to Run

The notebook was built in Google Colab. To run it:

1. Open `Project1_Renee-Guldi.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter.
2. Run all cells. The notebook downloads the dataset automatically using `gdown`.
     - Alternatively, download `creditcard.csv` from Kaggle and place it in the same folder as the notebook.
3. If running locally, install dependencies first: `pip install torch pandas numpy matplotlib gdown`

## Files

- `Project1_Renee-Guldi.ipynb`: data loading, preprocessing, model training, and evaluation
- `Guldi-Renee_Project1-Presentation.pptx`: project presentation slides

## Acknowledgments

Data-splitting and scaling helper functions were provided as part of CSCI-460 course materials.
