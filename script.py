import cv2 as cv
import numpy as np
import argparse
import os

# Nomes arquivos
BLUR = "blur.jpg"
BORDAS = "bordas.jpg"
GRADIENTE = "gradiente.jpg"
OUTPUT = "output.jpg"

paths = [
    BLUR,
    BORDAS,
    GRADIENTE,
    OUTPUT
]

def concatenar_paths(base_name):
    """
    Concatena os caminhos das imagens com o nome base fornecido.
    Args:
        base_name (str): Nome base para as imagens.
    Returns:
        list: Lista de caminhos concatenados.
    """
    return [os.path.join("Imagens testadas", f"{base_name}_{p}") for p in paths]

def processar_varias_imagens(pasta, janela):
    for nome_arquivo in os.listdir(pasta):
        if nome_arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            caminho = os.path.join(pasta, nome_arquivo)
            find(caminho, janela)

def detectar_circulos(img_colorida, img_bordas, output_img):
    """
    Encontra e desenha círculos na imagem de saída.
    """
    # 1. Transformada de Hough para círculos
    circles = cv.HoughCircles(
        img_bordas,
        cv.HOUGH_GRADIENT,
        dp=1,
        minDist=40,
        param1=100,
        param2=32,
        minRadius=5,
        maxRadius=40
    )

    # 2. Desenho dos círculos detectados
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            # Desenha o círculo
            cv.circle(output_img, (x, y), r, (0, 255, 0), 2)
            # Desenha o centro do círculo
            cv.circle(output_img, (x, y), 2, (0, 0, 255), 3)
        
        print(f"Detectados {len(circles[0])} círculo(s).")
    else:
        print("Nenhum círculo detectado.")

def detectar_poligonos(img_colorida, img_bordas, output_img):
    """
    Encontra, analisa e desenha polígonos (triângulos, retângulos, octógonos)
    que se assemelham a placas de trânsito.
    """
    # 1. Encontrar todos os contornos na imagem de bordas
    # cv.RETR_EXTERNAL é útil para pegar apenas os contornos externos
    contours, _ = cv.findContours(img_bordas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 2. Preparar máscaras de cor para validação (vermelho e amarelo)
    hsv = cv.cvtColor(img_colorida, cv.COLOR_BGR2HSV)
    
    # Vermelho (para placas PARE, Dê a Preferência, etc.)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask_r1 = cv.inRange(hsv, lower_red1, upper_red1)
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask_r2 = cv.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask_r1 + mask_r2

    # Amarelo (para placas de advertência)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # 3. Iterar sobre cada contorno encontrado
    for cnt in contours:
        # Filtro por área mínima para remover ruído
        area = cv.contourArea(cnt)
        if area < 250:  # Este valor pode precisar de ajuste
            continue

        # Aproxima o contorno para um polígono
        perimetro = cv.arcLength(cnt, True)
        # O segundo parâmetro (epsilon) controla a precisão da aproximação.
        # Um valor comum é 2-4% do perímetro.
        epsilon = 0.03 * perimetro 
        approx = cv.approxPolyDP(cnt, epsilon, True)

        # Obter a caixa delimitadora para analisar a cor
        x, y, w, h = cv.boundingRect(approx)

        # Identificar a forma pelo número de vértices
        num_vertices = len(approx)

        forma_detectada = ""
        cor_validada = False

        # É um octógono? (Placa PARE)
        if num_vertices == 8:
            # Validar pela cor vermelha
            roi = red_mask[y:y+h, x:x+w]
            # Se mais de 50% da área do contorno for vermelha
            if cv.countNonZero(roi) / area > 0.5:
                forma_detectada = "PARE (Octogono)"
                cor_validada = True

        # É um quadrado/retângulo? (Placas de Advertência)
        elif num_vertices == 4:
            # Validar pela cor amarela
            roi = yellow_mask[y:y+h, x:x+w]
            if cv.countNonZero(roi) / area > 0.5:
                forma_detectada = "Advertencia (Retangulo)"
                cor_validada = True
        
        # É um triângulo? (Placa Dê a Preferência)
        # NOTA: A placa "Dê a preferência" é um triângulo invertido. A detecção pode
        # ser mais complexa, mas vamos começar com a forma básica.
        elif num_vertices == 3:
             # Validar pela cor vermelha (da borda)
            roi = red_mask[y:y+h, x:x+w]
            # Para bordas, a porcentagem de cor será menor
            if cv.countNonZero(roi) / area > 0.1:
                forma_detectada = "Preferencia (Triangulo)"
                cor_validada = True

        # Se a forma e a cor foram validadas, desenhe na imagem de saída
        # if cor_validada:
        print(f"Detectado polígono: {forma_detectada}")
        # Desenha o contorno
        cv.drawContours(output_img, [approx], -1, (255, 0, 255), 3)
        # Escreve o nome da forma
        cv.putText(output_img, forma_detectada, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

def segmentar_por_cor(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Intervalos de Cor (Vermelho, Amarelo, Azul)
    # Vermelho
    lower_red1 = np.array([0, 120, 70]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70]); upper_red2 = np.array([180, 255, 255])
    red_mask = cv.bitwise_or(cv.inRange(hsv, lower_red1, upper_red1), cv.inRange(hsv, lower_red2, upper_red2))

    # Amarelo
    lower_yellow = np.array([20, 100, 100]); upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv.inRange(hsv, lower_yellow, upper_yellow)

    # Azul (placas de serviço/informação)
    lower_blue = np.array([100, 150, 0]); upper_blue = np.array([140, 255, 255])
    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Unifica todas as máscaras de cor em uma só
    final_color_mask = cv.bitwise_or(red_mask, yellow_mask)
    final_color_mask = cv.bitwise_or(final_color_mask, blue_mask)
    
    # Aplica um desfoque na máscara para suavizar e fechar pequenos buracos
    final_color_mask = cv.GaussianBlur(final_color_mask, (5, 5), 0)

    return blue_mask

def segmentar_por_morfologia(img):
    return

def find(img_path, janela):
    # 1. Leitura
    img = cv.imread(img_path)

    output = img.copy()

    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem em {img_path}")
    
    # 2. Conversão para escala de cinza, equalização de histograma para melhorar contraste e suavização
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    equ = cv.equalizeHist(gray)

    blur = cv.GaussianBlur(equ, (15, 15), 0)

    # 3. Aplicar operador de Sobel (X e Y)
    gx = cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=3)

    # 4. Magnitude do gradiente e binarização
    #    Usamos cv.magnitude para calcular a magnitude do gradiente
    #    e cv.threshold para binarizar a imagem resultante
    #    O valor 34 foi escolhido empiricamente para destacar as bordas
    grad = cv.magnitude(gx, gy)
    _, grad_thresh = cv.threshold(grad, 34, 255, cv.THRESH_BINARY_INV)

    # 5. Segmentação por cor (Vermelho, Amarelo, Azul)
    #    Usamos a função segmentar_por_cor para criar uma máscara de cor
    cor_mask = segmentar_por_cor(img)
    
    # 5. Detecção de bordas (Canny)
    # edges = cv.Canny(np.uint8(grad_thresh), 120, 240)
    
    # ========== Detecção de Círculos ==========
    #    Usamos a Transformada de Hough para detectar círculos
    
    # 6. Transformada de Hough para círculos
    # detectar_circulos(img, edges, output)
    
    # ========== Detecção de polígonos que podem ser placas ========== 
    # detectar_poligonos(img, edges, output)
    
    # 8. Exibe janela com o gradiente e com os círculos
    if janela == 1:
        cv.imshow("Imagem", img)
        # cv.imshow("Bordas", edges)
        cv.imshow("Máscara", cor_mask)
        # cv.imshow("Gradiente", grad_thresh)
        # cv.imshow("Gaussian", blur)
        # cv.imshow("Circulos Detectados", output)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    # 9. Salva as imagens processadas
    if not os.path.exists("Imagens testadas"):
        os.makedirs("Imagens testadas", exist_ok=True)

    paths_concatenadas = concatenar_paths(os.path.basename(img_path))

    # cv.imwrite(paths_concatenadas[0], blur)
    # cv.imwrite(paths_concatenadas[1], edges)
    # cv.imwrite(paths_concatenadas[2], grad_thresh)
    cv.imwrite(paths_concatenadas[3], output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de detecção de placas sem IA")
    parser.add_argument("imagem", help="Caminho para o arquivo de imagem")
    
    args = parser.parse_args()

    janela = int(input("Quer ver janelas? (1 = sim, 0 = não): "))

    if os.path.isdir(args.imagem):
        processar_varias_imagens(args.imagem, janela)
    else:
        find(args.imagem, janela)
