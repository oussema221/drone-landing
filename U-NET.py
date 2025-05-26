# Importation des bibliothèques
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

#  Définir les chemins vers les images
data_path = '/kaggle/input/semantic-drone-dataset'
org_imgs_path = os.path.join(data_path, 'dataset', 'semantic_drone_dataset', 'original_images')
masks_path = os.path.join(data_path, 'RGB_color_image_masks', 'RGB_color_image_masks')

#  2. Créer le modèle U-Net  
def my_unet(input_size=(256, 256, 3), n_classes=24):
    inputs = Input(input_size)
    
    # encodeur
    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

   
    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Milieu
    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)

    # decodeur
    u4 = UpSampling2D((2, 2))(c3)
    u4 = Conv2D(64, 2, activation='relu', padding='same')(u4)
    m4 = concatenate([c2, u4])
    c4 = Conv2D(64, 3, activation='relu', padding='same')(m4)
    c4 = Conv2D(64, 3, activation='relu', padding='same')(c4)

    u5 = UpSampling2D((2, 2))(c4)
    u5 = Conv2D(32, 2, activation='relu', padding='same')(u5)
    m5 = concatenate([c1, u5])
    c5 = Conv2D(32, 3, activation='relu', padding='same')(m5)
    c5 = Conv2D(32, 3, activation='relu', padding='same')(c5)

    # Sortie avec softmax (classification pixel par pixel)
    outputs = Conv2D(n_classes, 1, activation='softmax')(c5)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

#  Lire le fichier des couleurs/classes
csv_path = os.path.join(data_path, 'class_dict_seg.csv')
classes = pd.read_csv(csv_path)
cls2rgb = {cl: list(values) for cl, values in zip(classes['name'], classes.iloc[:, 1:].values)}

#  Transformer chaque masque couleur en masque "one-hot" 
def adjust_mask(mask):
    semantic_map = []
    for colour in cls2rgb.values():
        match = np.all(mask == colour, axis=-1)
        semantic_map.append(match)
    return np.float32(np.stack(semantic_map, axis=-1))

#  Générateur d'images et de masques
batch_size = 8
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
mask_datagen = ImageDataGenerator(validation_split=0.2)

def create_generator(image_dir, mask_dir, subset):
    img_gen = train_datagen.flow_from_directory(
        os.path.dirname(image_dir),
        classes=[os.path.basename(image_dir)],
        class_mode=None,
        seed=42,
        batch_size=batch_size,
        target_size=(256, 256),
        subset=subset
    )

    mask_gen = mask_datagen.flow_from_directory(
        os.path.dirname(mask_dir),
        classes=[os.path.basename(mask_dir)],
        class_mode=None,
        seed=42,
        batch_size=batch_size,
        target_size=(256, 256),
        color_mode='rgb',
        subset=subset
    )

    while True:
        imgs = next(img_gen)
        masks = adjust_mask(next(mask_gen))
        yield imgs, masks

train_gen = create_generator(org_imgs_path, masks_path, 'training')
val_gen = create_generator(org_imgs_path, masks_path, 'validation')

#  Entraîner le modèle
model = my_unet()

callbacks = [
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
]

steps_train = math.ceil(320 / batch_size)
steps_val = math.ceil(80 / batch_size)

history = model.fit(
    train_gen,
    steps_per_epoch=steps_train,
    validation_data=val_gen,
    validation_steps=steps_val,
    epochs=50,
    callbacks=callbacks
)

# Courbes de précision et de perte 
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Précision')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend(['Entraînement', 'Validation'])

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Perte')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend(['Entraînement', 'Validation'])

plt.tight_layout()
plt.show()

# === 8. Évaluation finale ===
score = model.evaluate(val_gen, steps=steps_val)
print(f"Perte : {score[0]:.4f}")
print(f"Précision : {score[1]:.4f}")

# === 9. Sauvegarde du modèle ===
model.save('drone_segmentation_model.keras')
