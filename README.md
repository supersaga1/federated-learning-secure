# Federated Learning with Homomorphic Encryption

## Description
A privacy-preserving machine learning project that simulates federated learning across multiple hospitals. It uses Paillier homomorphic encryption for secure model parameter aggregation and AES (Fernet) for encrypted communication.

## Technologies Used
- Python
- Paillier Encryption (`phe` library)
- AES Encryption (`cryptography`)
- Logistic Regression (`scikit-learn`)

## How to Run
1. Install dependencies:
```
pip install -r requirements.txt
```

2. Place your hospital CSV files in a `data/` folder with names:
- `hospital_1.csv`
- `hospital_2.csv`
- `hospital_3.csv`

3. Run the script:
```
python federated_learning_secure.py
```

## Note
This project is intended for academic use to demonstrate secure federated learning principles.
