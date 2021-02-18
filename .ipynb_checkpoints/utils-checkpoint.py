import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import nltk
import pickle

from sklearn.model_selection import train_test_split

def train_dev_test(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=seed)
    return X_train, X_dev, X_test, y_train, y_dev, y_test

def plot_train_acc(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plot_epochs = range(1, len(acc) + 1)

    plt.plot(plot_epochs, acc, 'r', label='Training acc')
    plt.plot(plot_epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def plot_train_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plot_epochs = range(1, len(loss) + 1)

    plt.plot(plot_epochs, loss, 'r', label='Training loss')
    plt.plot(plot_epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

with open('data/test_ids.pickle', 'rb') as t:
    X_id = pickle.load(t)
t.close()

def classifier_out(outputs, filename):
    out_rows=list(zip(X_id, outputs))
    out_rows = [('Id','Predicted')] + out_rows
    out_rows = [f'{t[0]},{t[1]}' for t in out_rows]
    with open(f'submissions/{filename}.csv', 'w') as a:
        a.write('\n'.join(out_rows))
    a.close()
