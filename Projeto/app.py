from flask import Flask, render_template, request
import cv2
import numpy as np
from scipy import ndimage

app = Flask(__name__)

INPUT_IMAGE = "static/pessoa.jpg"
OUTPUT_IMAGE = "static/output.jpg"


def filtro_moda(img, k):
    def moda(vetor):
        vetor = vetor.astype(np.uint8)
        hist = np.bincount(vetor)
        return np.argmax(hist)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mod = ndimage.generic_filter(img_gray, moda, size=k)
    return cv2.cvtColor(mod, cv2.COLOR_GRAY2BGR)


def aplicar_filtro(filtro, k):

    img = cv2.imread(INPUT_IMAGE)

    if img is None:
        print("Erro ao abrir a imagem.")
        return

    if k % 2 == 0:
        k += 1

    if filtro == "media":
        img_filtrada = cv2.blur(img, (k, k))

    elif filtro == "gauss":
        img_filtrada = cv2.GaussianBlur(img, (k, k), 0)

    elif filtro == "mediana":
        img_filtrada = cv2.medianBlur(img, k)

    elif filtro == "moda":
        img_filtrada = filtro_moda(img, k)

    elif filtro == "laplaciano":
        lap = cv2.Laplacian(img, cv2.CV_64F, ksize=k)
        img_filtrada = np.uint8(np.absolute(lap))


    else:
        img_filtrada = img

    cv2.imwrite(OUTPUT_IMAGE, img_filtrada)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/processar", methods=["POST"])
def processar():
    filtro = request.form.get("filtro")
    kernel = int(request.form.get("kernel"))

    aplicar_filtro(filtro, kernel)

    return render_template("index.html", filtro=filtro, kernel=kernel)


if __name__ == "__main__":
    app.run()
