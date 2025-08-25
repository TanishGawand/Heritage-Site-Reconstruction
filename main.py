import argparse
from image_loader import load_images
from feature_extractor import extract_features
from feature_matcher import match_features
from reconstruction_3d import reconstruct_3d
from point_cloud_export import save_point_cloud, visualize_with_matplotlib
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Convert 2D images to 3D model')
    parser.add_argument('--input', type=str, required=True, help='Path to folder containing images')
    args = parser.parse_args()
    
    print("=== 2D to 3D Reconstruction Pipeline ===")
    
    # Step 1: Load images
    print("\n1. Loading images...")
    images, image_paths = load_images(args.input)
    if len(images) < 2:
        print("Need at least 2 images for 3D reconstruction!")
        return
    
    # Step 2: Extract features
    print("\n2. Extracting features...")
    keypoints_list, descriptors_list = extract_features(images)
    
    # Step 3: Match features
    print("\n3. Matching features across images...")
    all_matches = match_features(descriptors_list)
    
    # Step 4: 3D Reconstruction
    print("\n4. Performing 3D reconstruction...")
    # Estimate camera intrinsic matrix (simplified assumption)
    h, w = images[0].shape[:2]
    focal_length = w * 1.2  # Rough estimate
    K = np.array([[focal_length, 0, w/2],
                  [0, focal_length, h/2],
                  [0, 0, 1]])
    
    points_3d, camera_poses = reconstruct_3d(keypoints_list, all_matches, K)
    
    if points_3d.size == 0:
        print("Reconstruction failed. Try images with more overlap and texture.")
        return
    
    # Step 5: Save and visualize results
    print("\n5. Saving results...")
    save_point_cloud(points_3d, "reconstructed_model.ply")
    visualize_with_matplotlib(points_3d)
    
    print(f"\n=== Pipeline Complete ===")
    print(f"Generated {len(points_3d)} 3D points")
    print("Output saved to 'reconstructed_model.ply'")

if __name__ == "__main__":
    main()
    