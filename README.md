# Detecção de Placas de Trânsito com Técnicas de Computação Visual

Este projeto tem como objetivo detectar placas de trânsito em imagens utilizando exclusivamente técnicas de computação visual, sem o uso de redes neurais ou aprendizado de máquina.

## Descrição

O sistema identifica e localiza placas de trânsito em imagens por meio de processamento de imagens, segmentação de cores, detecção de bordas e análise de formas geométricas.

## Tecnologias Utilizadas

- Python
- OpenCV

## Funcionalidades

- Pré-processamento de imagens
- Segmentação baseada em cor
- Detecção de bordas e contornos
- Identificação de regiões candidatas a placas
- Exibição dos resultados com as placas destacadas

## Como Executar

1. Clone este repositório.
2. Instale as dependências:

```bash
pip install opencv-python
```

3. Execute o script principal:

```bash
python main.py
```

## Estrutura do Projeto

- `main.py`: Script principal de execução.
- `utils/`: Funções auxiliares para processamento de imagem.
- `samples/`: Imagens de teste.

## Contribuição

Sinta-se à vontade para abrir issues ou enviar pull requests!

## Licença

Este projeto está sob a licença MIT.
