import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans

from utils.toHSV import cor_simples_hsv

def contornos_rect(contornos, coef=0.02):
    contour_list = []
    for cnt in contornos:
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, coef*peri, True)

        num_vertices = len(approx)
        if num_vertices == 4:
            contour_list.append(cnt)

    if not contour_list:
        return None
    else:
        # Ordena os contornos por área (decrescente) e pega os 5 maiores
        contour_list = sorted(contour_list, key=cv.contourArea, reverse=True)[:3]
        return contour_list

def detectar_circulos(img_colorida, img_bordas):
    """
    Encontra e desenha círculos na imagem de saída.
    """
    # Verificar se a imagem não está vazia e está no formato correto
    if img_bordas is None or img_bordas.size == 0:
        print("Erro: Imagem de bordas vazia ou None.")
        return None
    
    # Garantir que a imagem seja de canal único (escala de cinza)
    img_bordas = cv.cvtColor(img_bordas, cv.COLOR_BGR2GRAY) if len(img_bordas.shape) == 3 else img_bordas

    # cv.imshow("Imagem de bordas", img_disco)
    # cv.waitKey(0)

    img_bordas = cv.medianBlur(img_bordas, 5)  # Aplicar um desfoque para suavizar a imagem
    # 1. Transformada de Hough para círculos
    circles = cv.HoughCircles(
        img_bordas,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=200,
        param1=100,
        param2=18,
        minRadius=5,
        maxRadius=40
    )

    # 2. Desenho dos círculos detectados
    if circles is not None:
        circles = np.uint16(np.around(circles))

        imagens_cortadas = []
        
        # Iterar sobre cada círculo detectado
        for i in circles[0, :]:
            center_x, center_y, radius = i[0], i[1], i[2]
            
            # Calcular as coordenadas do retângulo envolvente (bounding box)
            x1 = max(0, center_x - radius)
            y1 = max(0, center_y - radius )
            x2 = min(img_colorida.shape[1], center_x + radius)
            y2 = min(img_colorida.shape[0], center_y + radius)
            
            # Realizar o corte (crop) da imagem ORIGINAL
            # Usamos a imagem original para manter as cores
            corte = img_colorida[y1:y2, x1:x2]
            
            # Adicionar o corte à lista se ele não for vazio
            if corte.size > 0:
                imagens_cortadas.append(corte)

                cv.imshow("Círculo detectado", corte)
                cv.waitKey(0)
        
        return imagens_cortadas, circles
    else:
        print("Nenhum círculo detectado.")
        return None, None

def extrair_retangulos(img_colorida, img_bordas):
    """
    Encontra, analisa e desenha polígonos (triângulos, retângulos, octógonos)
    que se assemelham a placas de trânsito.
    """
    imgs = []
    countours_list = []
    # 1. Encontrar todos os contornos na imagem de bordas
    # cv.RETR_EXTERNAL é útil para pegar apenas os contornos externos
    contours, _ = cv.findContours(img_bordas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 3. Iterar sobre cada contorno encontrado
    for cnt in contours:
        # Filtro por área mínima para remover ruído
        area = cv.contourArea(cnt)

        if area < 500:  # Este valor pode precisar de ajuste
            continue

        # Aproxima o contorno para um polígono
        perimetro = cv.arcLength(cnt, True)

        # O segundo parâmetro (epsilon) controla a precisão da aproximação.
        # Um valor comum é 2-4% do perímetro.
        epsilon = 0.02 * perimetro 
        approx = cv.approxPolyDP(cnt, epsilon, True)

        # Obter a caixa delimitadora para analisar a cor
        x, y, w, h = cv.boundingRect(cnt)

        height, width = img_colorida.shape[:2]

        placa_cortada = img_colorida[y:y+h, x:x+w]

        mascara = np.zeros(placa_cortada.shape[:2], dtype="uint8")

        contorno_local = cnt - (x, y)

        cv.drawContours(mascara, [contorno_local], -1, (255, 255, 255), -1) # O -1 preenche a forma

        placa_com_fundo_preto = cv.bitwise_and(placa_cortada, placa_cortada, mask=mascara)

        imgs.append(placa_com_fundo_preto)
        countours_list.append(approx)

    return imgs, countours_list

def canny(img, metodo, sigma=0.33):
    """
        Args:
        img: image
        method: Otsu, triangle, median and sobel
        sigma: 0.33 (default)
        2 outputs:
        edge_detection output, the high threshold for Hough Transform
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray, (15, 15), 0)

    if metodo=="median":
        thresh = np.median(blur)
        
    elif metodo=="triangle":
        thresh, _ = cv.threshold(blur, 0, 255, cv.THRESH_TRIANGLE)
        
    elif metodo=="otsu":
        thresh, _ = cv.threshold(blur, 0, 255, cv.THRESH_OTSU)
    
    elif metodo=="sobel":
        gx = cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=3)
        gy = cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=3)

        grad = cv.magnitude(gx, gy)
        thresh, _ = cv.threshold(grad, 34, 255, cv.THRESH_BINARY_INV)
        
    else:
        raise Exception("method specified not available!")

    lowTh = (1-sigma) * thresh
    highTh = (1+sigma) * thresh

    edges = cv.Canny(gray, lowTh, highTh)

    return edges, highTh

def extracao(img):
    # Convertendo para escala de cinza e binarizando.
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, bin_img = cv.threshold(gray, 5, 255, cv.THRESH_BINARY)

    # Detectando contornos e selecionando o maior.
    contornos, _ = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return [0, 0, 0, 'outra', 'outra']
        
    contorno_principal = max(contornos, key=cv.contourArea)

    # Calculando circularidade.
    area = cv.contourArea(contorno_principal)
    perimetro = cv.arcLength(contorno_principal, True)
    circularidade = (4 * np.pi * area) / (perimetro ** 2) if perimetro != 0 else 0

    # Calculando razao / largura.
    x, y, w, h = cv.boundingRect(contorno_principal)
    aspect_ratio = w / h if h != 0 else 0

    # Aproximando vértices.
    epsilon = 0.02 * perimetro
    aproximado = cv.approxPolyDP(contorno_principal, epsilon, True)
    num_vertices = len(aproximado)

    # Identificando cores principais.
    pixels = img[bin_img == 255]

    if len(pixels) < 2:
        return [circularidade, aspect_ratio, num_vertices, 'outra', 'outra']

    pixels_reshape = pixels.reshape(-1, 3).astype(np.float32)
    kmeans = KMeans(n_clusters=2, n_init='auto')
    kmeans.fit(pixels_reshape)
    cores_rgb = kmeans.cluster_centers_.astype(int)

    # kmeans.labels_ é um array que diz a qual cluster (0 ou 1) cada pixel pertence.
    # Vamos contar quantos pixels existem em cada cluster.
    counts = np.bincount(kmeans.labels_)

    # O índice do cluster com mais pixels é o da cor principal (fundo).
    idx_principal = np.argmax(counts)
    # O índice do cluster com menos pixels é o da cor secundária.
    idx_secundario = 1 - idx_principal # Se não for 0 é 1, e vice-versa

    cor_principal = cor_simples_hsv(cores_rgb[idx_principal])
    cor_secundaria = cor_simples_hsv(cores_rgb[idx_secundario])

    return [circularidade, aspect_ratio, num_vertices, cor_principal, cor_secundaria]
