#!/usr/bin/env python3
"""
Test Complete Data Processing Pipeline
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

def test_pipeline():
    print("🎯 Testing Complete Data Processing Pipeline")
    print("=" * 55)
    
    try:
        # Import our modules
        from kits23_loader import KITS23Loader
        from data_preprocessor import KITS23Preprocessor
        
        print("✅ Modules imported successfully")
        
        # Initialize loader and preprocessor
        loader = KITS23Loader()
        preprocessor = KITS23Preprocessor()
        
        # Test with first few cases
        test_cases = [0, 1, 2]
        successful = 0
        
        for case_id in test_cases:
            print(f"\n--- Testing Case {case_id} ---")
            
            # 1. Load data
            raw_volume = loader.load_volume(case_id)
            if raw_volume is None:
                print(f"   ❌ Failed to load case {case_id}")
                continue
            
            # 2. Preprocess data
            try:
                processed_tensor = preprocessor.preprocess(raw_volume)
                
                # Verify results
                assert processed_tensor.shape == (1, 128, 128, 128), f"Wrong shape: {processed_tensor.shape}"
                assert 0.0 <= processed_tensor.min() <= processed_tensor.max() <= 1.0, "Wrong value range"
                
                print(f"   ✅ Case {case_id} processed successfully")
                successful += 1
                
            except Exception as e:
                print(f"   ❌ Preprocessing failed for case {case_id}: {e}")
        
        print(f"\n📊 Pipeline Test Results:")
        print(f"   Successful: {successful}/{len(test_cases)} cases")
        
        if successful >= 2:
            print("🎉 Pipeline is working correctly!")
            return True
        else:
            print("❌ Pipeline needs debugging")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_pipeline()
    
    if success:
        print("\n🚀 Next step: Start model training!")
        print("   Run: python start_training.py")
    else:
        print("\n🔧 Please debug the pipeline first")
