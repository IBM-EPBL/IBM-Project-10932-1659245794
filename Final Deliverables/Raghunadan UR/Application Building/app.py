import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from flask import Flask, render_template, request
from flask import Flask, render_template

model = load_model("models\mnist.h5")

app = Flask(__name__)


def predictRes():
    global model
    img = Image.open("./static/result.png").convert("L")
    img = img.resize((28, 28))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 28, 28, 1)
    y_pred = model.predict(im2arr)
    re = list(y_pred[0]).index(max(y_pred[0]))
    plt.bar(list(range(10)), y_pred[0], align="center")
    plt.xticks(list(range(10)), list(range(10)))
    plt.xlabel("Digits")
    plt.ylabel("Accuracy")
    plt.savefig('./static/graph.png')
    plt.clf()
    plt.close()
    return re


@app.route('/')
def home():
    return render_template('./index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save("./static/result.png")
        res = predictRes()
        return render_template('./main.html', showcase=str(res))


if __name__ == '__main__':
    app.run()
