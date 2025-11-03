# nn.py
# Implementação de rede neural simples com uma camada oculta e backpropagation
# Autor: Mateus Ribeiro Fernandes
# Disciplina: Inteligência Artificial
# Profa: Cristiane Neri Nobre

import numpy as np

class SimpleNN:
    def __init__(self, n_in, n_hidden, n_out, lr=0.5, activation='sigmoid'):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr
        rng = np.random.default_rng(seed=42)

        # Inicialização Xavier/He simplificada
        self.W1 = rng.normal(0, 1, size=(n_hidden, n_in)) * np.sqrt(2 / (n_in + n_hidden))
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = rng.normal(0, 1, size=(n_out, n_hidden)) * np.sqrt(2 / (n_hidden + n_out))
        self.b2 = np.zeros((n_out, 1))
        self.activation = activation

    # Funções de ativação e derivadas
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return (x > 0).astype(float)

    # Forward propagation
    def forward(self, x):
        z1 = self.W1.dot(x) + self.b1
        if self.activation == 'sigmoid':
            a1 = self.sigmoid(z1)
        else:
            a1 = self.relu(z1)
        z2 = self.W2.dot(a1) + self.b2
        a2 = self.sigmoid(z2)
        cache = (x, z1, a1, z2, a2)
        return a2, cache

    # Backpropagation
    def backward(self, cache, y):
        x, z1, a1, z2, a2 = cache

        # Erro na saída
        delta2 = (a2 - y) * self.sigmoid_prime(z2)
        dW2 = delta2.dot(a1.T)
        db2 = np.sum(delta2, axis=1, keepdims=True)

        # Erro na camada oculta
        if self.activation == 'sigmoid':
            delta1 = self.W2.T.dot(delta2) * self.sigmoid_prime(z1)
        else:
            delta1 = self.W2.T.dot(delta2) * self.relu_prime(z1)
        dW1 = delta1.dot(x.T)
        db1 = np.sum(delta1, axis=1, keepdims=True)

        # Atualização de pesos
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    # Treinamento
    def train(self, X, Y, epochs=1000, verbose=False):
        losses = []
        for e in range(epochs):
            epoch_loss = 0
            for i in range(X.shape[1]):
                x = X[:, [i]]
                y = Y[:, [i]]
                a2, cache = self.forward(x)
                loss = 0.5 * np.sum((y - a2) ** 2)
                epoch_loss += loss
                self.backward(cache, y)
            losses.append(epoch_loss)
            if verbose and (e % 100 == 0 or e == epochs - 1):
                print(f"Epoch {e}: loss={epoch_loss:.4f}")
        return losses

    def predict(self, X):
        Y_pred = []
        for i in range(X.shape[1]):
            x = X[:, [i]]
            a2, _ = self.forward(x)
            Y_pred.append(a2.flatten())
        return np.array(Y_pred).T
