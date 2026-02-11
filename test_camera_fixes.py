"""Test script to verify camera image decoding fixes.

This script tests:
1. Invalid camera ID handling - should log clear error and return black image
2. Valid camera ID - should return actual image data
"""

import sys
sys.path.insert(0, 'e:/SimWorld')

from simworld.communicator.unrealcv import UnrealCV
from simworld.communicator.communicator import Communicator
import numpy as np

def test_camera_fixes():
    """Test the camera image decoding fixes."""
    print("=" * 60)
    print("Testing Camera Image Decoding Fixes")
    print("=" * 60)
    
    try:
        # Initialize UnrealCV
        print("\n1. Connecting to UnrealCV...")
        unrealcv_client = UnrealCV(port=9000, ip='127.0.0.1', resolution=(640, 480))
        communicator = Communicator(unrealcv=unrealcv_client)
        print("   ✓ Connected successfully")
        
        # Test 1: Invalid camera ID
        print("\n2. Testing invalid camera ID (999)...")
        try:
            img = communicator.get_camera_observation(999, 'lit', mode='direct')
            if img is not None:
                is_black = img.max() == 0
                print(f"   ✓ Returned fallback image: shape={img.shape}, is_black={is_black}")
                if is_black:
                    print("   ✓ Correctly returned black fallback image")
                else:
                    print("   ⚠ Warning: Image is not black, might be valid data")
            else:
                print("   ✗ Returned None instead of fallback image")
        except Exception as e:
            print(f"   ✗ Unexpected exception: {e}")
        
        # Test 2: Valid camera ID (assuming camera 0 exists)
        print("\n3. Testing valid camera ID (0)...")
        try:
            img = communicator.get_camera_observation(0, 'lit', mode='direct')
            if img is not None:
                is_black = img.max() == 0
                print(f"   ✓ Returned image: shape={img.shape}, is_black={is_black}")
                if not is_black:
                    print("   ✓ Successfully captured non-black image")
                else:
                    print("   ⚠ Warning: Image is black - camera might not be initialized or positioned correctly")
            else:
                print("   ✗ Returned None")
        except Exception as e:
            print(f"   ⚠ Exception (camera might not exist): {e}")
        
        # Test 3: Check camera ID assignment
        print("\n4. Testing camera ID assignment...")
        from simworld.agent.humanoid import Humanoid
        from simworld.utils.vector import Vector
        
        # Reset counters for clean test
        Humanoid._id_counter = 0
        Humanoid._camera_id_counter = 0
        
        h1 = Humanoid(Vector(0, 0), Vector(1, 0))
        h2 = Humanoid(Vector(10, 10), Vector(1, 0))
        h3 = Humanoid(Vector(20, 20), Vector(1, 0))
        
        print(f"   Humanoid 1: id={h1.id}, camera_id={h1.camera_id}")
        print(f"   Humanoid 2: id={h2.id}, camera_id={h2.camera_id}")
        print(f"   Humanoid 3: id={h3.id}, camera_id={h3.camera_id}")
        
        # Verify IDs are sequential
        if h1.id == 0 and h1.camera_id == 0:
            print("   ✓ Humanoid 1 IDs correct")
        else:
            print(f"   ✗ Humanoid 1 IDs incorrect: expected (0, 0), got ({h1.id}, {h1.camera_id})")
            
        if h2.id == 1 and h2.camera_id == 1:
            print("   ✓ Humanoid 2 IDs correct")
        else:
            print(f"   ✗ Humanoid 2 IDs incorrect: expected (1, 1), got ({h2.id}, {h2.camera_id})")
            
        if h3.id == 2 and h3.camera_id == 2:
            print("   ✓ Humanoid 3 IDs correct")
        else:
            print(f"   ✗ Humanoid 3 IDs incorrect: expected (2, 2), got ({h3.id}, {h3.camera_id})")
        
        print("\n" + "=" * 60)
        print("Testing Complete!")
        print("=" * 60)
        
        # Cleanup
        unrealcv_client.disconnect()
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_camera_fixes()
