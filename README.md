# Symbol Recognition Application

This project consists of two Python scripts that work together to create a machine learning-based symbol recognition application. The application generates a synthetic dataset of arrow symbols, trains a convolutional neural network (CNN) to recognize them, and provides a graphical user interface (GUI) for users to draw and predict symbols.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating and Training the Model](#generating-and-training-the-model)
  - [Running the GUI Application](#running-the-gui-application)
- [File Structure](#file-structure)
- [How It Works](#how-it-works)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)

## Overview

The application recognizes four arrow symbols: →, ↔, ←, and ↓. It uses a TensorFlow CNN model trained on a synthetic dataset of these symbols. The GUI, built with Tkinter, allows users to draw symbols on a canvas and predict them using the trained model.

## Features

- Generates a synthetic dataset of arrow symbols with variations (rotation, etc.).
- Trains a CNN model to classify the symbols with high accuracy.
- Provides a Tkinter-based GUI for drawing symbols and predicting them in real-time.
- Displays prediction confidence for each drawn symbol.
- Includes options to clear the canvas and re-draw.

## Requirements

- Python 3.6+
  
Libraries:
- tensorflow (for model training and prediction)
- numpy (for numerical operations)
- pillow (for image processing)
- matplotlib (for dataset visualization)
- scikit-learn (for dataset splitting)
- tkinter (for GUI, usually included with Python)

## Installation

1. Clone or download this repository to your local machine.
2. Install the required dependencies using pip:

    ```bash
    pip install tensorflow numpy pillow matplotlib scikit-learn
    ```

   Note: Tkinter is typically included with Python. If not, install it:
   - On Ubuntu/Debian: `sudo apt-get install python3-tk`
   - On macOS: Tkinter is usually pre-installed with Python.
   - On Windows: Tkinter is included with standard Python installations.

3. Ensure you have a working Python environment with the above libraries.

## Usage

### Generating and Training the Model

1. Run the `symbol_recognition.py` script to generate the dataset and train the CNN model:

    ```bash
    python symbol_recognition.py
    ```

   This script will:
   - Create a dataset directory with synthetic images of the four symbols.
   - Train the CNN model and save it as `symbol_recognition_model.h5`.
   - Display a plot of the generated symbols and training/validation accuracy.
   - Print the test accuracy and predictions for sample images.

### Running the GUI Application

1. Ensure the `symbol_recognition_model.h5` file exists in the same directory as the scripts.
2. Run the `test_prediction.py` script to start the GUI:

    ```bash
    python test_prediction.py
    ```

   The GUI will open with a canvas and two buttons:
   - **Draw**: Click and drag on the canvas to draw a symbol.
   - **Prever**: Click to predict the drawn symbol and display the result with confidence.
   - **Limpar**: Click to clear the canvas and start over.

## File Structure

```
symbol_recognition/
├── test_prediction.py              # Script for the Tkinter GUI application
├── symbol_recognition_model.py   # Script to generate dataset and train the model
├── symbol_recognition_model.h5  # Trained CNN model (generated after running train_model.py)
├── dataset/                    # Directory for synthetic dataset (generated after running train_model.py)
│   ├── →/                      # Images for right arrow
│   ├── ↔/                      # Images for double arrow
│   ├── ←/                      # Images for left arrow
│   └── ↓/                      # Images for down arrow
└── README.md                   # This file
```

## How It Works

### Dataset Generation (`symbol_recognition.py`):
- Creates 1000 synthetic images per symbol (→, ↔, ←, ↓) using PIL.
- Applies random rotations to add variation.
- Saves images in a dataset directory.

### Model Training (`symbol_recognition.py`):
- Loads and preprocesses the dataset (normalizes pixel values, reshapes for CNN).
- Splits data into training (80%) and testing (20%) sets.
- Builds a CNN with convolutional, pooling, and dense layers.
- Uses data augmentation (rotation, zoom, shifts) to improve robustness.
- Trains the model for 15 epochs and saves it as `symbol_recognition_model.h5`.

### GUI Application (`test_prediction.py`):
- Loads the trained model.
- Provides a 280x280 canvas for drawing with the mouse.
- Converts the drawn image to 28x28 (matching the training data) and predicts the symbol.
- Displays the predicted symbol and confidence in a message box.

## Limitations

- The model is trained on synthetic data, so it may struggle with hand-drawn symbols that differ significantly from the generated dataset.
- Only four symbols are supported (→, ↔, ←, ↓).
- The GUI drawing is basic and may not capture fine details of complex drawings.
- The model may misclassify symbols if the drawing is too noisy or ambiguous.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and test thoroughly.
4. Submit a pull request with a clear description of your changes.

Suggestions for improvements:
- Add support for more symbols.
- Enhance the GUI with features like undo/redo or adjustable brush size.
- Improve the model with real hand-drawn data.
- Optimize the CNN architecture for better accuracy.

![Imagem do WhatsApp de 2025-04-28 à(s) 12 42 23_7458b7ea](https://github.com/user-attachments/assets/a55743eb-ddfb-4ce1-ab84-791494028061)



![Imagem do WhatsApp de 2025-04-28 à(s) 12 43 32_61310263](https://github.com/user-attachments/assets/67d0858e-d92d-4785-8ab8-f7ea02365475)



![Imagem do WhatsApp de 2025-04-28 à(s) 12 46 23_e5609a58](https://github.com/user-attachments/assets/65d08f85-60b7-48d4-a299-2d0afb42903c)



![Imagem do WhatsApp de 2025-04-28 à(s) 13 14 57_7685c39f](https://github.com/user-attachments/assets/ab07a922-f0f4-41a9-8f4e-d6b77e028eee)


<img width="953" alt="Captura de tela 2025-04-28 092342" src="https://github.com/user-attachments/assets/c4991d7b-f257-43b6-b149-f5b276a6bf40" />


## License

This project is licensed under the MIT License. See the LICENSE file for details.
