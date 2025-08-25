import cv2
import os

def load_images(image_folder):
    """
    Loads all images from a specified folder.
    Returns a list of images and their file paths.
    """
    images = []
    image_paths = []
    
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    for filename in sorted(os.listdir(image_folder)):
        if filename.lower().endswith(valid_extensions):
            path = os.path.join(image_folder, filename)
            img = cv2.imread(path)
            if img is not None:
                # Resize for faster processing (optional but recommended)
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                images.append(img)
                image_paths.append(path)
                print(f"Loaded: {filename}")
    
    print(f"Successfully loaded {len(images)} images.")
    return images, image_paths