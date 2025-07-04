import cv2 as cv
import numpy as np

def cor_simples_hsv(rgb):
    """Converte uma cor BGR para HSV e a classifica de forma robusta."""
    # Converte um único pixel BGR para HSV
    hsv_color = cv.cvtColor(np.uint8([[rgb]]), cv.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv_color

    # Limiares para Saturação e Valor (Brilho)
    # Ignora cores com baixa saturação (cinzas) ou baixo valor (muito escuras)
    if s < 60 or v < 50:
        if v > 180: return 'branco'
        if v < 50: return 'preto'
        return 'cinza'

    # Agora classifica pela Matiz (Hue), que é a cor pura
    if h < 10 or h > 170:
        return 'vermelho'
    elif 15 < h < 40:
        return 'amarelo' # Faixa mais ampla para amarelo/laranja
    elif 100 < h < 130:
        return 'azul'
    else:
        return 'outra'