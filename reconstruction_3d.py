import numpy as np
import cv2

def reconstruct_3d(keypoints_list, all_matches, K):
    """
    The heart of the pipeline. Performs SfM and triangulation.
    K: Camera intrinsic matrix (assumed to be the same for all images)
    """
    # Placeholder for our 3D points (world coordinates)
    points_3d = []
    point_colors = []
    
    # Placeholder for camera poses. First camera is at origin.
    camera_poses = [np.eye(4)]
    
    # We'll use the first two images to initialize the 3D scene
    if len(all_matches) > 0:
        # Get matches between image0 and image1
        matches = all_matches[0]
        kp1 = keypoints_list[0]
        kp2 = keypoints_list[1]
        
        if len(matches) > 50:  # Ensure we have enough matches
            # Extract matched points
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
            
            # Find essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            # Recover relative camera pose (rotation and translation)
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
            
            # Create projection matrices
            # Camera 1: P1 = K [I | 0]
            P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            P1 = K.dot(P1)
            
            # Camera 2: P2 = K [R | t]
            P2 = np.hstack((R, t))
            P2 = K.dot(P2)
            
            # Triangulate points (convert to homogeneous coordinates)
            pts1 = pts1[mask.ravel() == 1]
            pts2 = pts2[mask.ravel() == 1]
            
            points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            points_3d = points_4d[:3] / points_4d[3]  # Convert to 3D
            
            # Store the second camera's pose
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t.ravel()
            camera_poses.append(pose)
            
            print(f"Triangulated {points_3d.shape[1]} 3D points from first two images.")
            
            # For simplicity, we'll just return these points
            # A full pipeline would continue with more images using bundle adjustment
            return points_3d.T, camera_poses
            
    print("Not enough matches for 3D reconstruction.")
    return np.array([]), []