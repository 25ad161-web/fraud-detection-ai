# Dataset folder

Place the Kaggle "Credit Card Fraud Detection" dataset here as `creditcard.csv`.

Download from: https://www.kaggle.com/datasets/mlg-ulb/ulb-machine-learning-group/creditcardfraud

Expected schema (31 columns):
- Time   : seconds elapsed between this transaction and the first transaction
- V1-V28 : PCA-anonymized numerical features
- Amount : transaction amount
- Class  : 1 = fraud, 0 = legit (target)

If `creditcard.csv` is not found, `scripts/generate_sample_data.py` will create a
statistically similar synthetic dataset (same schema, same ~0.17% fraud rate) so
the rest of the pipeline (EDA, training, API, frontend) can be run and tested
immediately without needing a Kaggle account.
