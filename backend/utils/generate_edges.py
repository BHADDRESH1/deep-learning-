import cv2
import os
import glob

def generate_edge_maps(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        
    images = glob.glob(os.path.join(source_folder, "*.jpg"))
    print(f"Found {len(images)} images in {source_folder}. Generating edge maps...")
    
    for img_path in images:
        basename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        # Blur to reduce noise (simulating 'ancient' photo noise)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Invert edges (black on white looks better for 'sketches')
        edges_inverted = cv2.bitwise_not(edges)
        
        output_path = os.path.join(target_folder, basename)
        cv2.imwrite(output_path, edges_inverted)
        
    print(f"Edge maps saved to {target_folder}")

def create_edge_map(input_path, output_path):
    """
    Generates an edge map for a single image (used by Flask App).
    """
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_inverted = cv2.bitwise_not(edges)
    cv2.imwrite(output_path, edges_inverted)
    return output_path

if __name__ == "__main__":
    generate_edge_maps("dataset/train", "dataset/edges")
