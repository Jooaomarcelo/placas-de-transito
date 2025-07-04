import cv2 as cv
import numpy as np
import argparse
import os

import utils.preProcess as preProcess
import utils.extracaoCaracteristicas as extracaoCaracteristicas
import utils.classificacao as classificacao

def processar_varias_imagens(pasta, segmentacao, janela):
    for nome_arquivo in os.listdir(pasta):
        if nome_arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
            caminho = os.path.join(pasta, nome_arquivo)
            find(caminho, segmentacao, janela)



def find(img_path, segmentacao="cor", janela=1):
    """
    Função para encontrar e desenhar contornos retangulares em uma imagem.
    Args:
        img_path: Caminho para a imagem a ser processada.
        segmentacao: Método de segmentação a ser utilizado ("cor" ou "bordas").
        janela: Se 1, exibe janelas de visualização; se 0, não exibe janelas.
    Returns:
        None: A função exibe a imagem com os contornos desenhados e não retorna nada.
    """
    img = cv.imread(img_path, cv.IMREAD_UNCHANGED)

    output = img.copy()

    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem em {img_path}")

    img = preProcess.melhorar_contraste(img)

    seg = None
    if segmentacao == "cor":
        print("Segmentação por cor")
        seg = preProcess.segmentar_por_cor(img)

    elif segmentacao == "bordas":
        seg = preProcess.segmentar_por_bordas(img)

    # Verificar se a segmentação foi bem-sucedida
    if seg is None:
        print("Erro: Método de segmentação não reconhecido ou falhou.")
        return
    
    # Verificar se a imagem segmentada não está vazia
    if seg.size == 0:
        print("Erro: Imagem segmentada está vazia.")
        return

    contornos, _ = cv.findContours(seg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # rects = extracaoCaracteristicas.contornos_rect(contornos, coef=0.02)
    imgs, rects = extracaoCaracteristicas.extrair_retangulos(img, seg)

    # Inicializar imgs como lista vazia se for None
    if imgs is None:
        imgs = []

    if rects is not None:
        cv.drawContours(output, rects, -1, (0, 255, 0), 2)
    else:
        print("Nenhum contorno retangular detectado.")

    # Verificar se a imagem segmentada é adequada para detecção de círculos
    if seg.dtype != np.uint8:
        seg = seg.astype(np.uint8)
    
    img_circles, circles = extracaoCaracteristicas.detectar_circulos(img, seg)

    if circles is not None:  
        imgs.append(img_circles)
            # Espera até que uma tecla seja pressionada

        for (x, y, r) in circles[0, :]:
            # Desenha o círculo
            cv.circle(output, (x, y), r, (0, 255, 0), 2)
            # Desenha o centro do círculo
            cv.circle(output, (x, y), 2, (0, 0, 255), 3)
    
    for img in imgs:
        res = extracaoCaracteristicas.extracao(img)

        placa = classificacao.classificar(res)

        if placa != 0:
            print(f"Placa detectada: {placa}")
            cv.imshow("Placa detectada", img)
            cv.waitKey(0)
    
    if janela:
        cv.imshow("Segmentaçao", seg)
        cv.imshow("Placas detectados", output)
        cv.waitKey(0)
    else:
        if not os.path.exists("output"):
            os.makedirs("output")
        
        base_path = os.path.splitext(os.path.basename(img_path))[0]
        base_path = os.path.join("output", base_path + ".png")
        cv.imwrite(base_path, output)
        # print(f"Imagem salva como {base_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de detecção de placas sem IA")
    parser.add_argument("imagem", help="Caminho para o arquivo de imagem")
    
    args = parser.parse_args()

    janela = int(input("Quer ver janelas? (1 = sim, 0 = não): "))
    # janela = 1

    segmentacao = "cor"

    if os.path.isdir(args.imagem):
        processar_varias_imagens(args.imagem, segmentacao, janela)
    else:
        find(args.imagem, segmentacao, janela)
