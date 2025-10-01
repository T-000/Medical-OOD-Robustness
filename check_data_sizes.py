import nibabel as nib
import numpy as np
import os
import sys

def analyze_folder(folder_path):
    print(f"Analyzing: {folder_path}")
    shapes = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.endswith('.nii.gz') or file.endswith('.nii'):
                    img = nib.load(file_path)
                    data = img.get_fdata()
                    shapes.append(data.shape)
                    print(f"  {file}: {data.shape}")
                elif file.endswith('.npz'):
                    npz = np.load(file_path)
                    if 'vol' in npz:
                        print(f"  {file}: {npz['vol'].shape}")
                    elif 'data' in npz:
                        print(f"  {file}: {npz['data'].shape}")
                    else:
                        print(f"  {file}: {list(npz.keys())}")
            except Exception as e:
                print(f"  {file}: ERROR - {e}")
    
    if shapes:
        print(f"\n=== SUMMARY ===")
        print(f"Total volumetric files: {len(shapes)}")
        print(f"Unique shapes: {set(shapes)}")

if __name__ == "__main__":
    target_folder = sys.argv[1] if len(sys.argv) > 1 else "."
    analyze_folder(target_folder)
