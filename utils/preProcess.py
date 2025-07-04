import cv2 as cv
import numpy as np

def segmentar_por_bordas(img):
    # 1. Converter a imagem para escala de cinza
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. Equalizar o histograma para melhorar o contraste
    equ = cv.equalizeHist(gray)

    # 3. Aplicamos um desfoque gaussiano para suavizar a imagem
    blur = cv.GaussianBlur(equ, (19, 19), 0)

    # 4. Aplicar operador de Sobel (X e Y)
    gx = cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=3)

    # 5. Magnitude do gradiente e binarização
    #    Usamos cv.magnitude para calcular a magnitude do gradiente
    #    e cv.threshold para binarizar a imagem resultante
    #    O valor 34 foi escolhido empiricamente para destacar as bordas
    grad = cv.magnitude(gx, gy)
    _, grad_thresh = cv.threshold(grad, 34, 255, cv.THRESH_BINARY_INV)

    # 6. Detecção de bordas (Canny)
    edges = cv.Canny(np.uint8(grad_thresh), 120, 240)

    return edges

def segmentar_por_cor(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Vermelho
    lower_red1 = np.array([0, 130, 90])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 130, 90])
    upper_red2 = np.array([180, 255, 255])

    # Amarelo
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
    
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)

    combined_mask = cv.bitwise_or(mask_red1, mask_red2)
    combined_mask = cv.bitwise_or(combined_mask, mask_yellow)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))

    combined_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel, iterations=1)

    return combined_mask

def melhorar_contraste(img):
    img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    L, a, b = cv.split(img_lab)
    L = cv.equalizeHist(L)
    img_lab_merge = cv.merge((L, a, b))
    return cv.cvtColor(img_lab_merge, cv.COLOR_Lab2BGR)
