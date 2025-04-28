# import numpy as np
# from PIL import Image, ImageDraw, ImageFont
# import os
# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # 1. Generate Synthetic Dataset
# symbols = ['→', '↔', '←', '↓']
# num_samples_per_symbol = 1000
# image_size = (28, 28)

# def generate_symbol_image(symbol, idx, save_path):
#     image = Image.new('L', image_size, color=255)
#     draw = ImageDraw.Draw(image)
#     try:
#         font = ImageFont.truetype("arial.ttf", 24)
#     except:
#         print(f"Fonte arial.ttf não encontrada para {symbol}, usando padrão")
#         font = ImageFont.load_default()
#     bbox = draw.textbbox((0, 0), symbol, font=font)
#     text_width = bbox[2] - bbox[0]
#     text_height = bbox[3] - bbox[1]
#     x = (image_size[0] - text_width) // 2
#     y = (image_size[1] - text_height) // 2 - 2  # Offset vertical
#     draw.text((x, y), symbol, fill=0, font=font)
    
#     # Adicionar variações (sem ruído)
#     image = image.rotate(np.random.uniform(-10, 10), fillcolor=255)
    
#     image = np.array(image)
#     # noise = np.random.normal(0, 2, image.shape).astype(np.uint8)  # Comentar ruído
#     # image = np.clip(image + noise, 0, 255).astype(np.uint8)
#     image = Image.fromarray(image)
#     image.save(os.path.join(save_path, f'{symbol}_{idx}.png'))

# # Apagar dataset antigo
# if os.path.exists('dataset'):
#     import shutil
#     shutil.rmtree('dataset')

# os.makedirs('dataset', exist_ok=True)
# for symbol in symbols:
#     symbol_path = os.path.join('dataset', symbol)
#     os.makedirs(symbol_path, exist_ok=True)
#     for i in range(num_samples_per_symbol):
#         generate_symbol_image(symbol, i, symbol_path)

# # Visualizar imagens geradas
# plt.figure(figsize=(8, 2))
# for i, symbol in enumerate(symbols):
#     img_path = f'dataset/{symbol}/{symbol}_0.png'
#     img = Image.open(img_path)
#     plt.subplot(1, len(symbols), i+1)
#     plt.imshow(img, cmap='gray')
#     plt.title(symbol)
#     plt.axis('off')
# plt.show()

# # 2. Load and Preprocess Dataset
# def load_dataset():
#     X, y = [], []
#     for idx, symbol in enumerate(symbols):
#         symbol_path = os.path.join('dataset', symbol)
#         for filename in os.listdir(symbol_path):
#             img = Image.open(os.path.join(symbol_path, filename)).convert('L')
#             img = np.array(img) / 255.0
#             X.append(img)
#             y.append(idx)
#     return np.array(X), np.array(y)

# X, y = load_dataset()
# X = X.reshape(-1, 28, 28, 1)

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 3. Build CNN Model
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(len(symbols), activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# # 4. Train the Model with Data Augmentation
# datagen = ImageDataGenerator(
#     rotation_range=10,
#     zoom_range=0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1
# )
# datagen.fit(X_train)

# history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
#                     epochs=15,
#                     validation_data=(X_test, y_test))

# # 5. Evaluate the Model
# test_loss, test_acc = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {test_acc:.4f}")

# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # 6. Save the Model
# model.save('symbol_recognition_model.h5')

# # 7. Test on Sample Images
# def predict_symbol(image_path):
#     img = Image.open(image_path).convert('L')
#     img = img.resize((28, 28), Image.Resampling.LANCZOS)
#     img = np.array(img) / 255.0
#     img = img.reshape(1, 28, 28, 1)
#     prediction = model.predict(img)
#     predicted_class = np.argmax(prediction)
#     return symbols[predicted_class]

# for symbol in symbols:
#     img_path = f'dataset/{symbol}/{symbol}_0.png'
#     predicted = predict_symbol(img_path)
#     print(f"Imagem: {symbol}, Previsto: {predicted}")



import numpy as np
from PIL import Image, ImageDraw
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. Generate Synthetic Dataset
symbols = ['→', '↔', '←', '↓']
num_samples_per_symbol = 1000
image_size = (28, 28)

def draw_symbol(image, symbol):
    draw = ImageDraw.Draw(image)
    if symbol == '→':
        draw.line((8, 14, 20, 14), fill=0, width=2)  # Linha horizontal
        draw.line((16, 10, 20, 14), fill=0, width=2)  # Ponta superior
        draw.line((16, 18, 20, 14), fill=0, width=2)  # Ponta inferior
    elif symbol == '↔':
        draw.line((8, 14, 20, 14), fill=0, width=2)  # Linha horizontal
        draw.line((12, 10, 8, 14), fill=0, width=2)  # Ponta esquerda superior
        draw.line((12, 18, 8, 14), fill=0, width=2)  # Ponta esquerda inferior
        draw.line((16, 10, 20, 14), fill=0, width=2)  # Ponta direita superior
        draw.line((16, 18, 20, 14), fill=0, width=2)  # Ponta direita inferior
    elif symbol == '←':
        draw.line((8, 14, 20, 14), fill=0, width=2)  # Linha horizontal
        draw.line((12, 10, 8, 14), fill=0, width=2)  # Ponta superior
        draw.line((12, 18, 8, 14), fill=0, width=2)  # Ponta inferior
    elif symbol == '↓':
        draw.line((14, 8, 14, 20), fill=0, width=2)  # Linha vertical
        draw.line((10, 16, 14, 20), fill=0, width=2)  # Ponta esquerda
        draw.line((18, 16, 14, 20), fill=0, width=2)  # Ponta direita
    return image

def generate_symbol_image(symbol, idx, save_path):
    image = Image.new('L', image_size, color=255)
    image = draw_symbol(image, symbol)
    
    # Adicionar variações
    image = image.rotate(np.random.uniform(-10, 10), fillcolor=255)
    
    image = np.array(image)
    image = Image.fromarray(image)
    image.save(os.path.join(save_path, f'{symbol}_{idx}.png'))

# Apagar dataset antigo
if os.path.exists('dataset'):
    import shutil
    shutil.rmtree('dataset')

os.makedirs('dataset', exist_ok=True)
for symbol in symbols:
    symbol_path = os.path.join('dataset', symbol)
    os.makedirs(symbol_path, exist_ok=True)
    for i in range(num_samples_per_symbol):
        generate_symbol_image(symbol, i, symbol_path)

# Visualizar imagens geradas
plt.figure(figsize=(8, 2))
for i, symbol in enumerate(symbols):
    img_path = f'dataset/{symbol}/{symbol}_0.png'
    img = Image.open(img_path)
    plt.subplot(1, len(symbols), i+1)
    plt.imshow(img, cmap='gray')
    plt.title(symbol)
    plt.axis('off')
plt.show()

# 2. Load and Preprocess Dataset
def load_dataset():
    X, y = [], []
    for idx, symbol in enumerate(symbols):
        symbol_path = os.path.join('dataset', symbol)
        for filename in os.listdir(symbol_path):
            img = Image.open(os.path.join(symbol_path, filename)).convert('L')
            img = np.array(img) / 255.0
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y)

X, y = load_dataset()
X = X.reshape(-1, 28, 28, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(symbols), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the Model with Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    epochs=15,
                    validation_data=(X_test, y_test))

# 5. Evaluate the Model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 6. Save the Model
model.save('symbol_recognition_model.h5')

# 7. Test on Sample Images
def predict_symbol(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return symbols[predicted_class]

for symbol in symbols:
    img_path = f'dataset/{symbol}/{symbol}_0.png'
    predicted = predict_symbol(img_path)
    print(f"Imagem: {symbol}, Previsto: {predicted}")
