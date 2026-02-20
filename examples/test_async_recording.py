import time
import os
from simworld.config import Config
from simworld.communicator.unrealcv import UnrealCV
from simworld.communicator.communicator import Communicator
from simworld.agent.humanoid import Humanoid
from simworld.utils.vector import Vector

def test_async_recording():
    print("🚀 Starting Async Recording Test...")
    
    # 1. Initialize
    config = Config()
    unrealcv = UnrealCV()
    communicator = Communicator(unrealcv)
    
    # 2. Spawn a humanoid
    pos = Vector(0, 0)
    dir = Vector(1, 0)
    humanoid = Humanoid(pos, dir)
    communicator.spawn_agent(humanoid, "TestHumanoid")
    time.sleep(2) # Wait for spawn
    
    # 3. Start background recording
    output_dir = "test_output"
    video_name = "test_movement.mp4"
    print(f"🎥 Starting background recording to {output_dir}/{video_name}...")
    communicator.start_background_recording(humanoid, output_dir=output_dir, video_name=video_name, fps=10)
    print(humanoid.id)
    
    # 4. Perform movements (main thread remains unblocked)
    print("🚶 Moving humanoid...")
    for i in range(20):
        communicator.humanoid_move_forward(humanoid.id)
        unrealcv.tick()
        time.sleep(1) # Simulate some work
    
    # 5. Stop recording
    print("⏹️ Stopping recording...")
    video_path = communicator.stop_background_recording(humanoid)
    
    # 6. Verify result
    if video_path and os.path.exists(video_path):
        print(f"✅ Success! Video saved at: {video_path}")
        print(f"📁 Files in {output_dir}: {os.listdir(output_dir)}")
    else:
        print("❌ Failed: Video file not found.")

if __name__ == "__main__":
    test_async_recording()
