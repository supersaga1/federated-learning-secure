import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from cryptography.fernet import Fernet
from phe import paillier
import pickle

hospital_files = [
    'data/hospital_1.csv',
    'data/hospital_2.csv',
    'data/hospital_3.csv'
]

def load_and_split_data(file_path, test_size=0.2):
    data = pd.read_csv(file_path)
    X = data[['blood_pressure', 'heart_rate']].values
    y = data['target'].values
    return train_test_split(X, y, test_size=test_size, random_state=42)

def train_local_model(X, y, init_weights, init_intercept):
    model = LogisticRegression(max_iter=500)
    model.coef_ = init_weights
    model.intercept_ = init_intercept
    model.classes_ = np.array([0, 1])
    model.fit(X, y)
    return model.coef_, model.intercept_

def encrypt_vector(vector, pubkey):
    return [pubkey.encrypt(float(v)) for v in vector.flatten()]

def decrypt_vector(encrypted_vector, privkey, shape):
    decrypted = [privkey.decrypt(v) for v in encrypted_vector]
    return np.array(decrypted).reshape(shape)

def aes_encrypt(data, key):
    f = Fernet(key)
    serialized = pickle.dumps(data)
    return f.encrypt(serialized)

def aes_decrypt(encrypted_data, key):
    f = Fernet(key)
    return pickle.loads(f.decrypt(encrypted_data))

def aggregate_encrypted_vectors(encrypted_vectors):
    num_clients = len(encrypted_vectors)
    summed = encrypted_vectors[0]
    for vec in encrypted_vectors[1:]:
        summed = [a + b for a, b in zip(summed, vec)]
    return [val / num_clients for val in summed]

def federated_learning(num_rounds=5):
    pubkey, privkey = paillier.generate_paillier_keypair()
    aes_keys = [Fernet.generate_key() for _ in hospital_files]

    global_weights = np.zeros((1, 2))
    global_intercept = np.zeros((1,))

    for rnd in range(num_rounds):
        encrypted_weights_list = []
        encrypted_intercepts_list = []

        for idx, file in enumerate(hospital_files):
            X_train, X_test, y_train, y_test = load_and_split_data(file)
            local_weights, local_intercept = train_local_model(X_train, y_train, global_weights, global_intercept)

            enc_weights = encrypt_vector(local_weights, pubkey)
            enc_intercepts = encrypt_vector(local_intercept, pubkey)

            aes_w = aes_encrypt(enc_weights, aes_keys[idx])
            aes_b = aes_encrypt(enc_intercepts, aes_keys[idx])

            encrypted_weights_list.append(aes_w)
            encrypted_intercepts_list.append(aes_b)

        decrypted_weights_list = [aes_decrypt(w, aes_keys[i]) for i, w in enumerate(encrypted_weights_list)]
        decrypted_intercepts_list = [aes_decrypt(b, aes_keys[i]) for i, b in enumerate(encrypted_intercepts_list)]

        aggregated_enc_weights = aggregate_encrypted_vectors(decrypted_weights_list)
        aggregated_enc_intercepts = aggregate_encrypted_vectors(decrypted_intercepts_list)

        global_weights = decrypt_vector(aggregated_enc_weights, privkey, (1, 2))
        global_intercept = decrypt_vector(aggregated_enc_intercepts, privkey, (1,))

        print(f"Round {rnd + 1}/{num_rounds} complete. Global weights and intercept updated securely.")

    final_model = LogisticRegression()
    final_model.coef_ = global_weights
    final_model.intercept_ = global_intercept
    final_model.classes_ = np.array([0, 1])

    blood_pressure = float(input("Enter blood pressure: "))
    heart_rate = float(input("Enter heart rate: "))
    user_input = np.array([[blood_pressure, heart_rate]])

    prediction = final_model.predict(user_input)
    print("Prediction:", "Disease" if prediction[0] == 1 else "No Disease")

if __name__ == '__main__':
    federated_learning()
