# Dynamic Retail Pricing Optimization Using ML and Cryptography

## Dataset
`retail_price.csv` (676 records, 30 columns).

## Project objective
Predict a retail product's unit price from historical product, sales, customer, calendar and competitor information. Apply a transparent demand-based adjustment and protect a price record using AES-256-GCM and SHA-256.

## Machine Learning
- Target: `unit_price`
- Model: Random Forest Regression
- Split: 80% training / 20% testing
- Metrics: MAE, RMSE, R²
- `total_price` is deliberately excluded because it is derived from `qty × unit_price`, which would leak the target.
- Inputs include product/category, quantity, freight, product attributes, customer count, calendar features, competitor prices/scores/freight and `lag_price`.

## Dynamic pricing logic
The prototype estimates demand intensity as `qty / customers`, compares it with the dataset median, and clips the multiplier to 0.85–1.15. The final price is:
`recommended_price = ML_predicted_unit_price × demand_multiplier`

The dataset does not include explicit inventory or supply availability, so this is a demand-indicator pricing rule, not a complete supply-demand optimization model.

## Cryptography
- AES-256-GCM: encrypts and authenticates the price record.
- SHA-256: creates a fingerprint to demonstrate record integrity.
- AES-GCM authentication detects ciphertext tampering. SHA-256 alone is not an authentication mechanism.
- The demo key is generated in Streamlit session memory. Production systems require secure key management.

## Run in VS Code
Open this folder in VS Code Terminal and run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Suggested 3-member work distribution
1. Member 1 – ML: dataset analysis, preprocessing, Random Forest, evaluation.
2. Member 2 – Dynamic pricing: demand indicator, multiplier, pricing scenarios and business logic.
3. Member 3 – Cryptography & integration: AES-GCM, SHA-256, Streamlit integration, testing and documentation.
