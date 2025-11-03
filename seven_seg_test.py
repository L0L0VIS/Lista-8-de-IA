# seven_seg_test.py
import numpy as np
from nn import SimpleNN

# Mapa de 7 segmentos para dígitos 0–9 (a,b,c,d,e,f,g)
segments = np.array([
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # a
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],  # b
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # c
    [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],  # d
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 1],  # e
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1],  # f
    [0, 0, 1, 1, 0, 1, 1, 0, 1, 1]   # g
])

# Codificação 4 bits (0–9)
def int_to_4bit(n):
    return np.array([(n >> 3) & 1, (n >> 2) & 1, (n >> 1) & 1, n & 1])

Y = np.column_stack([int_to_4bit(i) for i in range(10)])
X = segments.astype(float)

nn = SimpleNN(n_in=7, n_hidden=5, n_out=4, lr=0.5, activation='sigmoid')
losses = nn.train(X, Y, epochs=5000, verbose=True)

pred = nn.predict(X)
pred_bits = (pred > 0.5).astype(int)

print("Predições (bits):")
print(pred_bits)

# Avaliação simples
acc = np.mean(np.all(pred_bits == Y, axis=0)) * 100
print(f"Acurácia de treinamento: {acc:.2f}%")
