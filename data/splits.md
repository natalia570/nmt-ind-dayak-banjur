# Dataset Splits

This project uses manually collected parallel data for
Indonesian – Dayak Banjur neural machine translation (NMT).

## Data Distribution

The dataset is split into three subsets:

- **Training set**: 8,120 sentence pairs  
  Used to train the NMT model.

- **Validation set**: 2,030 sentence pairs  
  Used during training for model evaluation and hyperparameter tuning.

- **Test set**: 100 sentence pairs  
  Used exclusively for final BLEU score evaluation.

## Data Usage Policy

- Data augmentation techniques are applied **only to the training set**.
- Validation and test sets are kept unchanged to ensure fair evaluation.
- The test set is never used during training or model selection.

## Repository Note

Due to data ownership and size considerations, this repository only includes
small **sample datasets** for each split. The full dataset is available upon
reasonable request.
