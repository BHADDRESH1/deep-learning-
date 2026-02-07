import numpy as np
import os
import cv2
import glob
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 384
CHANNELS = 3
BATCH_SIZE = 4
EPOCHS = 10  # Low for demonstration; increase for real results
TRAIN_PATH = 'dataset/train'
WEIGHTS_PATH = 'weights/weights.weights.h5'

def load_data(path):
    image_files = glob.glob(os.path.join(path, "*.jpg"))
    data = []
    for f in image_files:
        img = cv2.imread(f)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0 # Normalize
        data.append(img)
    return np.array(data)

def create_mask(batch_size, img_height, img_width):
    # Create random rectangular masks (Context Encoder style)
    mask = np.ones((batch_size, img_height, img_width, 3), dtype=np.float32)
    for i in range(batch_size):
        # Random rectangle
        h = np.random.randint(40, 100)
        w = np.random.randint(40, 100)
        y = np.random.randint(0, img_height - h)
        x = np.random.randint(0, img_width - w)
        mask[i, y:y+h, x:x+w, :] = 0.0
    return mask

def build_model():
    input_img = Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))

    # Encoder
    x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    
    # Bottleneck (Fully Convolutional to save memory)
    # 32x48x256 -> 32x48x512
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    # No Flatten/Dense/Reshape needed


    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) # Output 0-1

    model = Model(input_img, x)
    model.compile(optimizer=Adam(learning_rate=0.0002), loss='mse')
    return model

def train():
    images = load_data(TRAIN_PATH)
    if len(images) == 0:
        print("No images found in dataset/train! Run generate_synthetic_data.py first.")
        return

    model = build_model()
    model.summary()
    
    print("Starting training...")
    
    # Simple training loop
    for epoch in range(EPOCHS):
        # Shuffle
        np.random.shuffle(images)
        
        # Batching
        for i in range(0, len(images), BATCH_SIZE):
            batch_imgs = images[i : i+BATCH_SIZE]
            if len(batch_imgs) < BATCH_SIZE: continue
            
            masks = create_mask(len(batch_imgs), IMG_HEIGHT, IMG_WIDTH)
            masked_imgs = batch_imgs * masks
            
            # Train: Match the masked input to the ORIGINAL image (reconstruction)
            loss = model.train_on_batch(masked_imgs, batch_imgs)
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.5f}")
        
    # Save weights
    if not os.path.exists('weights'):
        os.makedirs('weights')
    model.save_weights(WEIGHTS_PATH)
    print(f"Weights saved to {WEIGHTS_PATH}")

if __name__ == "__main__":
    train()
