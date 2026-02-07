import numpy as np
import cv2
import os

def create_synthetic_art(path, num_images=20, size=(256, 384)):
    if not os.path.exists(path):
        os.makedirs(path)
        
    print(f"Generating {num_images} synthetic images in {path}...")
    
    for i in range(num_images):
        # Create a blank image
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        
        # Add random background color (brownish/sepia types)
        bg_color = np.random.randint(100, 200, 3)
        img[:] = bg_color
        
        # Draw random shapes to simulate "art"
        for _ in range(10):
            color = np.random.randint(0, 255, 3).tolist()
            pt1 = (np.random.randint(0, size[1]), np.random.randint(0, size[0]))
            pt2 = (np.random.randint(0, size[1]), np.random.randint(0, size[0]))
            cv2.line(img, pt1, pt2, color, np.random.randint(1, 5))
            
            center = (np.random.randint(0, size[1]), np.random.randint(0, size[0]))
            radius = np.random.randint(5, 50)
            cv2.circle(img, center, radius, color, -1)
            
        # Save
        filename = os.path.join(path, f"art_{i}.jpg")
        cv2.imwrite(filename, img)
        
    print("Generation complete.")

if __name__ == "__main__":
    create_synthetic_art("dataset/train", num_images=20)
    create_synthetic_art("dataset/test", num_images=5)
