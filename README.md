# Neural Network From Scratch

Complete implementation of a neural network, built primarily with Python and NumPy, without Deep Learning frameworks.

## Overview

This project demonstrates a functional implementation of artificial neural networks, from the perceptron neuron to a full multilayer network trained with Stochastic Gradient Descent (SGD).

## Results

Based on the MNIST dataset, this network achieved approximately 97% accuracy in 50 epochs with a 0.1 learning rate.

## Installation and Setup

Follow the steps below to set up and run the project locally:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/ArthurDOli/Neural-Network-From-Scratch.git
    cd Neural-Network-From-Scratch
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python main.py
    ```

## Project Structure

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
