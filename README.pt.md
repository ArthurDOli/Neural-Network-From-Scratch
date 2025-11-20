# Rede Neural do Zero

Implementação completa de uma rede neural, construída principalmente com Python e NumPy, sem ferramentas de Deep Learning.

## Visão Geral

Este projeto demonstra uma implementação funcional de redes neurais artificiais, desde o neurônio perceptron até uma rede multicamadas completa treinada com Stochastic Gradient Descent (SGD).

## Resultados

Baseando-se no dataset MNIST, essa rede teve aproximadamente 97% de acurácia em 50 épocas com 0.1 de learning rate.

## Instalação e Configuração

Siga os passos abaixo para configurar e executar o projeto localmente:

1.  **Clone o repositório:**

    ```bash
    git clone https://github.com/ArthurDOli/Neural-Network-From-Scratch.git
    cd Neural-Network-From-Scratch
    ```

2.  **Crie e ative um ambiente virtual:**

    ```bash
    python -m venv venv
    # No Windows
    venv\Scripts\activate
    # No macOS/Linux
    source venv/bin/activate
    ```

3.  **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação:**
    ```bash
    python main.py
    ```

## Estrutura do Projeto

```bash
Neural-Network-From-Scratch/
├── src/
│ ├── cost_functions.py
│ ├── main.py
│ ├── network.py
│ └── neurons.py
├── .gitignore
├── LICENSE
├── README.md
├── README.pt.md
└── requirements.txt
```
