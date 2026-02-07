import cv2
import numpy as np
from PIL import Image, ImageEnhance

def enhance_image(image_path, output_path):
    """
    Applies lightweight post-processing to the GAN output to improve visual quality.
    
    Techniques:
    1. Mild Contrast Enhancement (using PIL ImageEnhance.Contrast)
    2. Subtle Sharpening (using Unsharp Masking via OpenCV)
    
    Args:
        image_path (str): Path to the input image (GAN output).
        output_path (str): Path to save the enhanced image.
        
    Returns:
        str: The output path of the enhanced image.
    """
    try:
        # Load image using PIL for color/contrast work
        img = Image.open(image_path)
        
        # 1. Contrast Enhancement
        # Factor 1.0 is original, 1.2 is 20% boost. Keeping it mild.
        contrast = ImageEnhance.Contrast(img)
        img_contrast = contrast.enhance(1.15) 
        
        # 2. Color/Saturation Enhancement (Optional but often good for "restoration" feel)
        # Bringing out the faded colors slightly
        color = ImageEnhance.Color(img_contrast)
        img_enhanced = color.enhance(1.1)
        
        # Convert to OpenCV format for sharpening (PIL -> numpy)
        img_np = np.array(img_enhanced)
        # RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # 3. Sharpening using Unsharp Masking
        # Gaussian Blur creates the "smooth" version
        gaussian = cv2.GaussianBlur(img_bgr, (0, 0), 3.0)
        # Weighted sum: Original + (Original - Smoothed) * amount
        # formula: src1 * alpha + src2 * beta + gamma
        # Unsharp mask formula: refined = original + (original - blurred) * amount
        # This implementation: cv2.addWeighted(original, 1.5, blurred, -0.5, 0) gives a sharpened look
        img_sharp = cv2.addWeighted(img_bgr, 1.4, gaussian, -0.4, 0)
        
        # Save output
        cv2.imwrite(output_path, img_sharp)
        
        return output_path
        
    except Exception as e:
        print(f"Error in post-processing: {e}")
        # If enhancement fails, just simplify copy original to output or return None
        # returning None to let caller handle it (or just save original)
        return None
