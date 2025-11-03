# xor_test.py
import numpy as np
from nn import SimpleNN

# Dados do problema XOR
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

X = X.astype(float)
Y = Y.astype(float)

nn = SimpleNN(n_in=2, n_hidden=2, n_out=1, lr=0.8, activation='sigmoid')
losses = nn.train(X, Y, epochs=5000, verbose=True)

pred = nn.predict(X)
print("Previsões finais (XOR):")
print(np.round(pred, 3))
