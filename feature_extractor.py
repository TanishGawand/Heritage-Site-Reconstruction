import cv2
import numpy as np

def extract_features(images):
    """
    Extracts keypoints and descriptors from all images using SIFT.
    Returns a list of keypoints and descriptors for each image.
    """
    sift = cv2.SIFT_create()
    keypoints_list = []
    descriptors_list = []
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)
        
        if des is None:
            des = np.array([])
            
        keypoints_list.append(kp)
        descriptors_list.append(des)
        print(f"Extracted {len(kp)} features from an image.")
    
    return keypoints_list, descriptors_list