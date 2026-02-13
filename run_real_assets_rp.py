
import sys
import os
import time
from unittest.mock import MagicMock

# Mock heavy dependencies
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['simworld.utils.data_importer'] = MagicMock()

# Mock GUI deps
sys.modules['pyqtgraph'] = MagicMock()
sys.modules['pyqtgraph.Qt'] = MagicMock()
sys.modules['PyQt5'] = MagicMock()
sys.modules['PyQt5.QtCore'] = MagicMock()
sys.modules['PyQt5.QtGui'] = MagicMock()
sys.modules['PyQt5.QtWidgets'] = MagicMock()

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simworld.assets_rp.AssetsRP import AssetsRetrieverPlacer
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV
from simworld.config.config_loader import Config

def main():
    print("Initializing connection to Unreal Engine SimWorld...")
    
    try:
        # 1. Initialize Real Communicator (Standard)
        # Assumes UE is running and listening on default port (9000)
        # Using the standard client because magic number is restored!
        unrealcv_client = UnrealCV() 
        communicator = Communicator(unrealcv_client)
        print("Connected to SimWorld.")
    except Exception as e:
        print(f"Failed to connect to SimWorld: {e}")
        print("Please ensure the SimWorld UE project is running.")
        return

    # 2. Initialize AssetsRetrieverPlacer
    dummy_config = Config()
    
    print("Initializing AssetsRetrieverPlacer...")
    assets_rp = AssetsRetrieverPlacer(
        config=dummy_config, 
        input_dir="dummy_path", 
        communicator=communicator
    )

    # Apple asset was missing. Using a standard Pedestrian asset for verification.
    asset_path = "/Game/TrafficSystem/Pedestrian/Base_Pedestrian.Base_Pedestrian_C"
    # Corrected Z to 620 based on demo_city_1 ground level analysis
    location = (0, 0, 620) 
    rotation = (0, 0, 0)
    scale = (5, 5, 5) # 5x Pedestrian (20x might cover the camera)

    print(f"Spawning Pedestrian at {location} with scale {scale}...")
    success = assets_rp.place_asset(asset_path, location, rotation, scale)
    
    if success:
        print("Spawn executed successfully. Check the simulation window for the Giant Apple.")
    else:
        print("Spawn failed. Check logs.")

if __name__ == "__main__":
    main()
