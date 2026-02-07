import numpy as np
import os
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from tensorflow.keras.optimizers import Adam

# Parameters match train_model.py
IMG_HEIGHT = 256
IMG_WIDTH = 384
CHANNELS = 3
WEIGHTS_PATH = 'weights/weights.weights.h5'

def build_model():
    # MUST MATCH train_model.py EXACTLY
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
    
    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    model = Model(input_img, x)
    return model

# Global model instance
_model = None

def get_model():
    global _model
    if _model is None:
        print("Loading model for the first time...")
        _model = build_model()
        if os.path.exists(WEIGHTS_PATH):
            _model.load_weights(WEIGHTS_PATH)
            print("Weights loaded.")
        else:
            print(f"WARNING: Weights not found at {WEIGHTS_PATH}")
    return _model

def restore_image(input_path, output_path):
    """
    Restores the image at input_path and saves to output_path.
    """
    model = get_model()
    
    # Load and preprocess
    original = cv2.imread(input_path)
    if original is None:
        raise ValueError("Could not load input image")
        
    # Resize for model
    img_resized = cv2.resize(original, (IMG_WIDTH, IMG_HEIGHT))
    input_data = img_resized / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    # Predict
    prediction = model.predict(input_data)
    restored = prediction[0]
    
    # Post-process
    restored_img = (restored * 255).astype(np.uint8)
    
    # Save (Resize back to original? Or keep model output size? 
    # Keeping model output size is safer for quality, or resize back if critical.
    # For now, saving 256x384 result.)
    cv2.imwrite(output_path, restored_img)
    
    return output_path
