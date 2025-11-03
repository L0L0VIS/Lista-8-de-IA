# run_experiments.py
# Roda experimentos para XOR e 7-seg, salva curvas de loss, previsões e logs
# Requisitos: numpy, matplotlib
# No Colab: já tem matplotlib e numpy instalados

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# --- Classe SimpleNN (igual ao nn.py) ---
class SimpleNN:
    def __init__(self, n_in, n_hidden, n_out, lr=0.5, activation='sigmoid', seed=42):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lr = lr
        rng = np.random.default_rng(seed=seed)
        self.W1 = rng.normal(0, 1, size=(n_hidden, n_in)) * np.sqrt(2 / (n_in + n_hidden))
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = rng.normal(0, 1, size=(n_out, n_hidden)) * np.sqrt(2 / (n_hidden + n_out))
        self.b2 = np.zeros((n_out, 1))
        self.activation = activation

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        z1 = self.W1.dot(x) + self.b1
        a1 = self.sigmoid(z1) if self.activation=='sigmoid' else self.relu(z1)
        z2 = self.W2.dot(a1) + self.b2
        a2 = self.sigmoid(z2)
        return a2, (x, z1, a1, z2, a2)

    def backward(self, cache, y):
        x, z1, a1, z2, a2 = cache
        delta2 = (a2 - y) * self.sigmoid_prime(z2)
        dW2 = delta2.dot(a1.T)
        db2 = np.sum(delta2, axis=1, keepdims=True)
        if self.activation=='sigmoid':
            delta1 = self.W2.T.dot(delta2) * self.sigmoid_prime(z1)
        else:
            delta1 = self.W2.T.dot(delta2) * self.relu_prime(z1)
        dW1 = delta1.dot(x.T)
        db1 = np.sum(delta1, axis=1, keepdims=True)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, Y, epochs=1000, verbose=False):
        losses = []
        for e in range(epochs):
            epoch_loss = 0.0
            for i in range(X.shape[1]):
                x = X[:, [i]]
                y = Y[:, [i]]
                a2, cache = self.forward(x)
                loss = 0.5 * np.sum((y - a2) ** 2)
                epoch_loss += loss
                self.backward(cache, y)
            losses.append(epoch_loss)
            if verbose and (e % 100 == 0 or e == epochs-1):
                print(f"Epoch {e} loss={epoch_loss:.6f}")
        return losses

    def predict(self, X):
        outs = []
        for i in range(X.shape[1]):
            x = X[:, [i]]
            a2, _ = self.forward(x)
            outs.append(a2.flatten())
        return np.array(outs).T

# --- Helpers para salvar imagens e logs ---
def save_loss_plot(losses, filename):
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.xlabel('Épocas')
    plt.ylabel('Loss (soma MSE por época)')
    plt.title('Curva de perda')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def save_predictions_table_text(inputs, expected, predicted, filename):
    # salva uma imagem com texto (table-like) para inserir no relatório
    txt = []
    for i in range(len(inputs)):
        txt.append(f"{inputs[i]} -> pred: {predicted[i]} (esperado: {expected[i]})")
    content = "\n".join(txt)
    with open(filename.replace('.png','.txt'), 'w') as f:
        f.write(content)
    # criar imagem simples com matplotlib
    plt.figure(figsize=(6, 0.4*len(txt)+0.5))
    plt.text(0, 1, content, fontsize=10, family='monospace')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

# --- Experimento XOR ---
def experiment_xor(outdir='results/xor'):
    ensure_dir(outdir)
    X = np.array([[0,0,1,1],[0,1,0,1]], dtype=float)
    Y = np.array([[0,1,1,0]], dtype=float)
    nn = SimpleNN(n_in=2, n_hidden=2, n_out=1, lr=0.8, activation='sigmoid', seed=123)
    losses = nn.train(X, Y, epochs=5000, verbose=True)
    save_loss_plot(losses, os.path.join(outdir,'xor_loss.png'))
    pred = nn.predict(X)  # shape (1,4) => transpose handling below
    pred_list = [float(round(v,4)) for v in pred.flatten()]
    exp_list = [int(v) for v in Y.flatten()]
    inputs = [[int(X[0,i]), int(X[1,i])] for i in range(X.shape[1])]
    save_predictions_table_text(inputs, exp_list, pred_list, os.path.join(outdir,'xor_preds.png'))
    # save numeric log
    with open(os.path.join(outdir,'xor_log.txt'),'w') as f:
        f.write(f"Final predictions: {pred_list}\\nLoss final: {losses[-1]}\\n")
    print("XOR done. Results in", outdir)

# --- Experimento 7-seg ---
def int_to_4bit(n):
    return np.array([(n>>3)&1, (n>>2)&1, (n>>1)&1, n&1])

def experiment_7seg(outdir='results/7seg'):
    ensure_dir(outdir)
    segments = np.array([
        [1,1,1,1,1,1,0,1,1,1],  # NOTE: arrange rows as digits later; we'll transpose to match cols
        [1,1,1,0,1,0,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1],
    ])
    # To be safe: use explicit table per digit (rows are segments)
    segments = np.array([
        [1,1,1,1,1,1,0,0,1,1], # a (adjusted)
        [1,1,1,1,1,0,0,1,1,1], # b
        [1,1,0,1,1,1,1,1,1,1], # c
        [1,0,1,1,0,1,1,0,1,1], # d
        [1,0,1,0,0,1,1,0,1,1], # e
        [1,0,1,1,0,1,1,1,1,1], # f
        [0,0,1,1,0,1,1,0,1,1]  # g
    ], dtype=float)
    X = segments.astype(float)  # shape (7,10)
    Y = np.column_stack([int_to_4bit(i) for i in range(10)])  # (4,10)

    nn = SimpleNN(n_in=7, n_hidden=5, n_out=4, lr=0.5, activation='sigmoid', seed=321)
    losses = nn.train(X, Y, epochs=5000, verbose=True)
    save_loss_plot(losses, os.path.join(outdir,'7seg_loss.png'))

    pred = nn.predict(X)  # shape (4,10) returned as (n_out, n_examples)
    pred_bits = (pred > 0.5).astype(int).T  # shape (10,4)
    expected_bits = Y.T.astype(int)  # (10,4)
    # prepare readable prints
    inputs = [list(X[:,i].astype(int)) for i in range(X.shape[1])]
    pred_list = [''.join(map(str,row.tolist())) for row in pred_bits]
    exp_list = [''.join(map(str,row.tolist())) for row in expected_bits]
    # save table image & txt
    save_predictions_table_text(inputs, exp_list, pred_list, os.path.join(outdir,'7seg_preds.png'))
    with open(os.path.join(outdir,'7seg_log.txt'),'w') as f:
        f.write("Pred bits (rows digit0..9):\\n")
        for i in range(10):
            f.write(f"digit {i}: inputs={inputs[i]} expected={exp_list[i]} pred={pred_list[i]}\\n")
        acc = np.mean([pred_list[i]==exp_list[i] for i in range(10)])*100
        f.write(f"Training accuracy: {acc:.2f}%\\n")
    print("7-seg done. Results in", outdir)

# --- main ---
if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_xor(outdir=f'results/xor_{timestamp}')
    experiment_7seg(outdir=f'results/7seg_{timestamp}')
    print("All experiments finished.")
