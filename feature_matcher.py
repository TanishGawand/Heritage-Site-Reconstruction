import cv2
import numpy as np

def match_features(descriptors_list):
    """
    Matches features between consecutive image pairs using FLANN.
    Returns matches for each image pair.
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    all_matches = []
    
    for i in range(len(descriptors_list) - 1):
        des1 = descriptors_list[i]
        des2 = descriptors_list[i + 1]
        
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            all_matches.append([])
            continue
            
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Apply Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        all_matches.append(good_matches)
        print(f"Matches between image {i} and {i+1}: {len(good_matches)}")
    
    return all_matches