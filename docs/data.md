# Data

The preferred dataset is IEEE-CIS: `train_transaction.csv` plus `train_identity.csv`.
The fallback is PaySim. Rows are sorted by `TransactionDT` and split 70% / 15% / 15%
for train, validation, and test.

No random split is used. Feature generation is point-in-time: each row only sees prior
transactions for the same account entity.
