def classificar(vetor):
    # Vetor: [circularidade, aspect radio, vertices, cor principal, cor secundaria].
    
    placa = 0   # Tipo da placa identificada, seguindo a lógica: [0: não é placa, 1: regulamentação, 2: advertência, 3: indicação, 4: sinalização temporária].

    if vetor[3] in ["vermelho", "branco"] and vetor[4] in ["vermelho", "branco"]:
        if vetor[0] >= 0.85 or (vetor[1] >= 0.75 and vetor[1] <= 1.25) or (vetor[2] == 3 or vetor[2] == 8):
            placa = 1
    elif vetor[3] == "amarelo":
        if (vetor[0] <= 0.5 and (vetor[1] >= 0.75 and vetor[1] <= 1.25)) or vetor[2] == 4:
            placa = 2
            
    return placa
