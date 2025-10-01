#!/usr/bin/env python3
"""
KiTS23 Data Loader - Complete Implementation
"""
import os
import nibabel as nib
import numpy as np

class KITS23Loader:
    def __init__(self, data_path="kits23/dataset"):
        self.data_path = data_path
        
    def load_volume(self, case_id):
        """Load 3D imaging data for a specific case"""
        case_dir = os.path.join(self.data_path, f"case_{case_id:05d}")
        image_path = os.path.join(case_dir, "imaging.nii.gz")
        
        print(f"🔍 Loading case {case_id}...")
        print(f"   Path: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Case {case_id} not found at: {image_path}")
            return None
        
        try:
            # Load NIfTI file
            image = nib.load(image_path)
            image_data = image.get_fdata()
            
            print(f"✅ Successfully loaded case {case_id}")
            print(f"   Shape: {image_data.shape}")
            print(f"   Data range: [{image_data.min():.1f}, {image_data.max():.1f}]")
            print(f"   Data type: {image_data.dtype}")
            
            return image_data
            
        except Exception as e:
            print(f"❌ Failed to load case {case_id}: {e}")
            return None
    
    def load_segmentation(self, case_id):
        """Load segmentation labels for a specific case"""
        case_dir = os.path.join(self.data_path, f"case_{case_id:05d}")
        seg_path = os.path.join(case_dir, "segmentation.nii.gz")
        
        if os.path.exists(seg_path):
            try:
                seg = nib.load(seg_path)
                seg_data = seg.get_fdata()
                unique_labels = np.unique(seg_data)
                print(f"   Segmentation labels: {unique_labels}")
                return seg_data
            except Exception as e:
                print(f"❌ Failed to load segmentation for case {case_id}: {e}")
                return None
        else:
            print(f"⚠️  Segmentation not found for case {case_id}")
            return None
    
    def get_available_cases(self):
        """Get list of all available case IDs"""
        available_cases = []
        for i in range(500):  # Check up to 500 cases
            case_dir = os.path.join(self.data_path, f"case_{i:05d}")
            if os.path.exists(case_dir):
                available_cases.append(i)
        return available_cases

# Test the loader
if __name__ == "__main__":
    loader = KITS23Loader()
    
    print("🎯 KiTS23 Data Loader Test")
    print("=" * 40)
    
    # Test loading a case
    test_case = 0
    volume = loader.load_volume(test_case)
    
    if volume is not None:
        segmentation = loader.load_segmentation(test_case)
        print(f"\n✅ Loader is working correctly!")
    else:
        print(f"\n❌ Please check if data is downloaded to: {loader.data_path}")
