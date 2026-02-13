
import sys
import os
import socket
import struct
import time

# Standard UnrealCV Client (Assuming environment is fixed)
from simworld.communicator.unrealcv import UnrealCV

def probe_asset(client, path, name_hint="Probe"):
    print(f"--- Probing: {path} ---")
    # Generate unique name
    unique_name = f"{name_hint}_{int(time.time())}"
    
    # Try to spawn
    cmd = f'vset /objects/spawn_bp_asset {path} {unique_name}'
    res = client.client.request(cmd)
    
    if "error" in res.lower() or "can not find" in res.lower():
        print(f"[MISSING] Server replied: {res}")
        return False
    else:
        print(f"[FOUND]   Server replied: {res}")
        # Clean up (optional, or leave it to see)
        # client.client.request(f'vset /objects/delete {unique_name}')
        # Set location just in case we want to see it
        client.client.request(f'vset /object/{unique_name}/location 0 0 620')
        client.client.request(f'vset /object/{unique_name}/scale 5 5 5')
        return True

def main():
    print("Connecting to SimWorld...")
    try:
        client = UnrealCV()
        print("Connected.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # List of paths to probe
    candidates = [
        # The path you want
        "/Game/ModularSciFi/SM_Apple/BP_Apple.BP_Apple_C",
        
        # Common variations/Guesses
        "/Game/ModularSciFi/Blueprints/SM_Apple.SM_Apple",
        "/Game/ModularSciFi/Meshes/SM_Apple.SM_Apple",
        "/Game/ModularSciFi/SM_Apple.SM_Apple"
        
    ]

    print("\nStarting Asset Probe...")
    print("Note: UnrealCV cannot 'list' folders. We can only check exact paths.")
    
    found_count = 0
    for path in candidates:
        if probe_asset(client, path):
            found_count += 1
            
    print(f"\nProbe Complete. Found {found_count}/{len(candidates)} assets.")

    return 0

if __name__ == "__main__":
    main()
