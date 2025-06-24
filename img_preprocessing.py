import cv2
import numpy as np
import os
from PIL import Image
import tempfile
from pathlib import Path

class MeterImagePreprocessor:
    """
    Advanced image preprocessing pipeline for smart meter OCR
    Optimized for LCD displays and seven-segment digits
    """
    
    def __init__(self):
        self.temp_files = []
    
    def normalize_image(self, image):
        """Step 1: Normalize pixel intensity values"""
        norm_img = np.zeros((image.shape[0], image.shape[1]))
        normalized = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    
    def deskew_image(self, image):
        """Step 2: Correct image skewness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Find all non-zero pixels (text pixels)
        coords = np.column_stack(np.where(gray > 0))
        
        # Get rotation angle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Correct angle calculation
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), 
                                flags=cv2.INTER_CUBIC, 
                                borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def enhance_resolution(self, image_path, target_dpi=300):
        """Step 3: Scale image to optimal resolution"""
        with Image.open(image_path) as img:
            # Calculate scaling factor
            length_x, width_y = img.size
            factor = max(1, float(1024.0 / max(length_x, width_y)))
            
            # Apply scaling
            new_size = (int(factor * length_x), int(factor * width_y))
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save with high DPI
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            temp_filename = temp_file.name
            img_resized.save(temp_filename, dpi=(target_dpi, target_dpi))
            self.temp_files.append(temp_filename)
            
            return temp_filename
    
    def remove_noise(self, image):
        """Step 4: Advanced noise removal for LCD displays"""
        if len(image.shape) == 3:
            # For color images
            denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)
        else:
            # For grayscale images
            denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        
        return denoised
    
    def enhance_contrast(self, image):
        """Step 5: Enhance contrast for LCD displays"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # For grayscale images
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def convert_to_grayscale(self, image):
        """Step 6: Convert to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def apply_adaptive_threshold(self, image):
        """Step 7: Apply adaptive thresholding"""
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(blurred, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Apply morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def preprocess_image(self, image_path, save_intermediate=False):
        """Complete preprocessing pipeline"""
        print(f"Processing: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Step 1: Enhance resolution
        high_res_path = self.enhance_resolution(image_path)
        image = cv2.imread(high_res_path)
        
        # Step 2: Normalize
        image = self.normalize_image(image)
        
        # Step 3: Deskew
        image = self.deskew_image(image)
        
        # Step 4: Remove noise
        image = self.remove_noise(image)
        
        # Step 5: Enhance contrast
        image = self.enhance_contrast(image)
        
        # Step 6: Convert to grayscale
        gray_image = self.convert_to_grayscale(image)
        
        # Step 7: Apply adaptive threshold
        binary_image = self.apply_adaptive_threshold(gray_image)
        
        if save_intermediate:
            base_name = Path(image_path).stem
            cv2.imwrite(f"{base_name}_processed.png", binary_image)
        
        return binary_image, gray_image
    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        self.temp_files.clear()

# Usage example
if __name__ == "__main__":
    preprocessor = MeterImagePreprocessor()
    
    # Process single image
    try:
        binary_img, gray_img = preprocessor.preprocess_image("images/000090225651470522210.png", save_intermediate=True)
        print("Preprocessing completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        preprocessor.cleanup()
