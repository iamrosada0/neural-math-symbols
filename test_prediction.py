import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model

# Carregar o modelo treinado
model = load_model('symbol_recognition_model.h5')
symbols = ['→', '↔', '←', '↓']

# Função para prever o símbolo
def predict_symbol():
    image = Image.new('L', (280, 280), color=255)
    draw = ImageDraw.Draw(image)
    for x, y in points:
        draw.ellipse((x-7, y-7, x+7, y+7), fill=0)  # Aumentar espessura
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image.save('drawn_symbol.png')
    img = np.array(image) / 255.0
    img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_symbol = symbols[predicted_index]
    confidence = np.max(prediction) * 100
    messagebox.showinfo("Previsão", f"Símbolo: {predicted_symbol}\nConfiança: {confidence:.2f}%")

# Função para limpar o canvas
def clear_canvas():
    canvas.delete("all")
    points.clear()

# Função de desenho
def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x-7, y-7, x+7, y+7, fill='black', outline='black')
    points.append((x, y))

# Criar a janela principal
root = tk.Tk()
root.title("Reconhecimento de Símbolos")

# Criar canvas
canvas = tk.Canvas(root, width=280, height=280, bg='white')
canvas.pack()

# Lista para armazenar pontos
points = []

# Bind para capturar eventos de desenho
canvas.bind("<B1-Motion>", draw)

# Botões
predict_button = tk.Button(root, text="Prever", command=predict_symbol)
predict_button.pack()
clear_button = tk.Button(root, text="Limpar", command=clear_canvas)
clear_button.pack()

# Iniciar interface
root.mainloop()
