import numpy as np

def save_point_cloud(points_3d, filename="output.ply"):
    """
    Saves the 3D point cloud to a PLY file (can be viewed in MeshLab, Blender, etc.)
    """
    if points_3d.size == 0:
        print("No points to save.")
        return
    
    header = f"""ply
format ascii 1.0
element vertex {len(points_3d)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # For now, assign a default color (white) to all points
    # In an advanced version, you would project image colors onto the 3D points
    with open(filename, 'w') as f:
        f.write(header)
        for point in points_3d:
            x, y, z = point
            f.write(f"{x} {y} {z} 255 255 255\n")
    
    print(f"Point cloud saved to {filename}")

def visualize_with_matplotlib(points_3d):
    """
    Simple 3D visualization using Matplotlib
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reconstructed 3D Point Cloud')
    
    # Equal aspect ratio
    max_range = np.array([points_3d[:, 0].max()-points_3d[:, 0].min(), 
                         points_3d[:, 1].max()-points_3d[:, 1].min(),
                         points_3d[:, 2].max()-points_3d[:, 2].min()]).max() / 2.0
    mid_x = (points_3d[:, 0].max()+points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max()+points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max()+points_3d[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.show()