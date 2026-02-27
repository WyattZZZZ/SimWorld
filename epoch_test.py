import os
import time
import math
import re
from typing import Dict, Optional
from PIL import Image
import json

# SimWorld Imports
from simworld.config import Config
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV
from simworld.llm.a2a_llm import A2ALLM
from simworld.map.map import Map
from simworld.agent.humanoid import Humanoid
from simworld.assets_rp.AssetsRP import AssetsRetrieverPlacer as AssetsRP

# temporary data
from examples.few_shot import few_shot_data


# Set API Key
os.environ["GEMINI_API_KEY"] = "AIzaSyC5_M3sNYEaiGD27Lu0Zozsb9ToV8TWOy0"
os.environ["OPENAI_API_KEY"] = "EMPTY"


# --- 0. Initialize Communication (Global) ---

print("Initializing UnrealCV Connection...")
unrealcv = UnrealCV()
communicator = Communicator(unrealcv)
config = Config()

# Test Connection
try:
    status = unrealcv.client.request("vget /unrealcv/status")
    unrealcv.client.request("vrun setinputmodeui")
    print(f"Connection Status: {status}")
except Exception as e:
    print(f"Connection Failed: {e}")
    print("Please ensure the SimWorld game is running!")

import math
import re

# --- 1. Define Vector Helper Class ---


class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Vector({self.x:.2f}, {self.y:.2f})"

    def distance(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


# --- 2. Define Agent Class (使用 A2ALLM 支持视觉) ---


class Agent:
    def __init__(self, role="hider", few_shot_data=None):
        self.llm = A2ALLM(
            "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
            provider="openai",
            url="http://localhost:8000/v1",
            few_shot_data=few_shot_data,
        )
        self.role = role
        self.last_action = None  # 记录上一步动作，供提示使用

        if role == "hider":
            self.system_prompt = """# Role: Hide and Seek Hider

You are currently playing **Hide and Seek** as the Hider. Your goal is to find an optimal outdoor hiding spot and secure it by placing an apple.

## 🛠 Constraints & Boundaries
- **Accessibility:** - **No Jumping:** Any spot requiring a jump is invalid.
- **No Buildings:** You must remain outdoors. Entering buildings is strictly prohibited.
- **Boarder**: You are also provided boarder, don't go out of the boarder, which means more than 100 meters away from the origin.
- **Turn Around**: If you find robot is not moving in video or cannot see anything, try rotate 180
- **Pay attention to Your steps**: Use it wisely, you only have 5 steps.

## Hint:
- **You are also provided apple_placed, if it is true, it means you have placed the apple, so you should hide yourself, which means running away from the apple.**
- **You must go as far as you can and try to find a corner, but you should not go out of the boarder.**


## 🕹 Available Actions
Choose and output **EXACTLY ONE** of the following:
* `forward <duration>` (between `0` and `30`)
* `rotate <angle> <direction>` (e.g., `rotate 135 right`)
* `place_apple` (Use when the spot is found, forbidden when apple_placed is true and Steps Left > 5)
* `wait`

## ⚠️ Critical Output Rule
- **Output ONLY the action command.**
- **You must choose to rotate once you find you are not moving in video while performing a forward action.**
- **Only place apple when you at last 3 steps.**
- **Only place 1 apple**: Once apple_placed is true, you can not use place_apple again.
"""
        else:  # seeker
            self.system_prompt = """# Role: Hide and Seek Seeker

Your objective is to locate a **hidden red apple** in the environment. 

## 🔍 Strategy & Intelligence
- **Visual Cues:** If you see a **Hider Robot**, look around carefully—the apple is likely nearby!
- **Target:** Use the `found` command only when the red apple is within your field of vision.
- **Sample:** You are provided some sample images in various distances of that apple, they are all captured in a empty environment, you can use them to estimate the distance.

## 🛠 Constraints & Boundaries
- **Accessibility:** - **No Jumping:** Areas requiring jumping are off-limits.
- **No Buildings:** You are strictly prohibited from entering any buildings.
- **Boarder**: You are also provided boarder, don't go out of the boarder, which means more than 100 meters away from the origin.
- **Turn Around**: If you find robot is not moving in video or cannot see anything, try rotate 180
- **Pay attention to Your steps**: Use it wisely, you only have 25 steps.
- **View**: You can use the `found` when you see the apple in your view of a non empty environment, and less than 20 meters away.
- **Found**: you must use it with one sentence, this sentence shows the approximate apple's position in your view.

## Hint:
- **Hider will only hide inside the boarder. And since he was only provided with 10 steps, he will hide as close to the origin as possible.**
- **Try to search those corners.**


## 🕹 Available Actions
Output **EXACTLY ONE** of the following commands:
1. `forward <duration>` (between `0` and `30`)
2. `rotate <angle> <direction>` (e.g., `rotate 135 left`)
3. `found <sentence>` (e.g., `found apple is in front of me` or `found apple is next to that building made of glass`) 
4. `wait` (Stay still for 1 second)

## ⚠️ Strict Output Rule
- **Output ONLY the action command.**
- **You must choose to rotate once you find you are not moving in video while performing a forward action.**
"""

    def action(self, obs, step=0, target=None, apple_placed=False):
        """Decide next action based on observation.

        Args:
            obs: Dictionary containing:
                - 'position' (Vector): Current 2D position
                - 'direction' (Vector): Current direction unit vector
                - 'ego_view' (ndarray): First-person camera view
                - 'video' (str): Path to video file
            target: Optional target position (Vector) for navigation hints

        Returns:
            str: Action command
        """
        position = obs["position"]
        direction = obs["direction"]
        ego_view = obs["ego_view"]
        video_path = obs["video"]

        # Convert direction vector to yaw angle
        current_yaw = math.degrees(math.atan2(direction.y, direction.x))

        if self.role == "hider" and target:
            # Calculate angle to target for hider
            delta_x = target.x - position.x
            delta_y = target.y - position.y
            target_angle = math.degrees(math.atan2(delta_y, delta_x))

            angle_diff = target_angle - current_yaw
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360

            prompt = f"""
Current position: {position}
Distance from origin (m): {int((position.x**2 + position.y**2) ** 0.5) / 100}
Current direction: {direction} (yaw: {current_yaw:.1f}°)
Last Action: {self.last_action if hasattr(self, "last_action") else "None"}
Steps Left: {step}
Apple_placed: {apple_placed}
"""
        else:  # seeker
            prompt = f"""
Current position: {position}
Distance from origin (m): {int((position.x**2 + position.y**2) ** 0.5) / 100}
Current direction: {direction} (yaw: {current_yaw:.1f}°)
Last Action: {self.last_action if hasattr(self, "last_action") else "None"}
Steps Left: {step}
"""

        # 使用 A2ALLM 的 generate_instructions 方法（支持图像）
        print(prompt)
        action, time = self.llm.generate_instructions(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            video_path=video_path,
            images=[ego_view] if ego_view is not None else [],
        )
        self.last_action = action  # 记录本次动作，供下次提示使用
        print(action)

        return action.strip().lower().strip('"').strip("'")


# --- 3. Define Environment Class ---


class HideAndSeekEnv:
    def __init__(self, communicator, config=None):
        """初始化环境（不需要 AssetsRP）"""
        self.communicator = communicator
        self.config = config or Config()
        self.map = Map(self.config)
        # 使用相对路径
        self.map.initialize_map_from_file(
            roads_file=os.path.join("../data/example_city/demo_city_1/roads.json")
        )

        self.agents = {}  # 例如 {'hider': Humanoid(...), 'seeker': Humanoid(...)}
        self.agent_names = {}  # {'hider': 'Base_User_Agent_C_0', ...}
        self.target = None
        self.apple_location = None
        self.apple_name = None  # 存储生成的苹果名称
        self.phase = "HIDING"  # HIDING or SEEKING

    def destroy(self):
        self.communicator.destroy_object(self.apple_name)
        for key, item in self.agent_names.items():
            self.communicator.destroy_humanoid(item)

    def reset(self, role="hider"):
        """Reset the environment and spawn/reset the humanoid agent.

        Args:
            role: 'hider' or 'seeker'

        Returns:
            observation: Dict with 'position', 'direction', 'ego_view'
        """
        agent_bp = "/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C"

        # Initial spawn position and facing direction
        spawn_location = Vector(0, 0)
        spawn_forward = Vector(1, 0)

        # First time: Create and spawn the agent
        self.agents[role] = Humanoid(
            position=spawn_location,
            direction=spawn_forward,
            map=self.map,
            communicator=self.communicator,
        )

        # Spawn the humanoid in Unreal
        self.communicator.spawn_agent(
            self.agents[role], name=None, model_path=agent_bp, type="humanoid"
        )
        self.communicator.humanoid_set_speed(self.agents[role].id, 200)

        # Cache the UE actor name
        self.agent_names[role] = self.communicator.get_humanoid_name(
            self.agents[role].id
        )

        # Set target hiding spot for hider
        if role == "hider":
            self.target = Vector(1700, -1700)
            self.phase = "HIDING"
        else:
            self.phase = "SEEKING"

        time.sleep(3)  # Wait for spawn

        # Attempt to bind the agent's camera to the UE camera closest to the actor
        try:
            if self.communicator is not None and self.agents[role] is not None:
                ue_name = self.communicator.get_humanoid_name(self.agents[role].id)
                found_cam = self.communicator.find_camera_id_for_actor(ue_name)
                if found_cam is not None:
                    self.agents[role].camera_id = found_cam
        except Exception:
            pass

        # Get initial observation
        return self.get_observation(role)

    def step(self, action, apple_placed=False, done=False, role="hider"):
        """Parse and execute the action.

        Returns:
            observation: Dict with 'position', 'direction', 'ego_view'
            reward: Negative distance to target (for hider) or 0
            done: Boolean indicating if episode is complete
            success: Boolean indicating if action was successfully parsed
        """
        action_cleaned = action.strip().strip('"').strip("'").lower()
        success = False
        video = None
        try:
            self.communicator.start_background_recording(self.agents[role], fps=10)
            print("Background recording started")
        except Exception as e:
            print("Failed to start background recording: ", e)

        if action_cleaned.startswith("forward"):
            match = re.search(r"forward\s+(\d+\.?\d*)", action_cleaned)
            if match:
                duration = float(match.group(1))
                self.communicator.humanoid_step_forward(
                    self.agents[role].id, duration, direction=0
                )
                success = True
            else:
                print(f"[Warning] Failed to parse: '{action}'")

        elif action_cleaned.startswith("rotate"):
            match = re.search(r"rotate\s+(\d+\.?\d*)\s+(left|right)", action_cleaned)
            if match:
                angle = float(match.group(1))
                direction = match.group(2)
                self.communicator.humanoid_rotate(
                    self.agents[role].id, angle, direction
                )
                success = True
            else:
                print(f"[Warning] Failed to parse: '{action}'")

        elif "place_apple" in action_cleaned and not apple_placed:
            video = self.communicator.stop_background_recording(self.agents[role])
            self.place_apple_at_agent()
            success = True
            apple_placed = True
            done = True

        elif "found" in action_cleaned:
            global sentence
            sentence = action_cleaned.split("found")[1].strip()
            print(sentence)
            print("[SUCCESS] Seeker found the apple!")
            success = True
            done = True

        elif "wait" in action_cleaned:
            time.sleep(1)
            success = True

        else:
            print(f"[Warning] Unknown action: '{action}'")
            time.sleep(0.5)

        if not done:
            video = self.communicator.stop_background_recording(self.agents[role])

        # Get current state from UE
        observation = self.get_observation(role, video)

        # Update agent internal state
        loc_3d = self.communicator.unrealcv.get_location(self.agent_names[role])
        orientation = self.communicator.unrealcv.get_orientation(self.agent_names[role])
        self.agents[role].position = Vector(loc_3d[0], loc_3d[1])
        self.agents[role].direction = orientation[1]  # yaw angle

        # Calculate reward
        reward = 0
        if self.phase == "HIDING" and self.target:
            reward = -self.agents[role].position.distance(self.target)

        return observation, reward, done, success, apple_placed

    def place_apple_at_agent(self, role="hider"):
        """使用直接 UnrealCV 命令在 agent 当前位置放置苹果"""
        loc = self.communicator.unrealcv.get_location(self.agent_names[role])

        # 苹果的蓝图路径（根据 probe_assets.py 中的路径）
        apple_bp = "/Game/ModularSciFi/BP_Apple/BP_Apple.BP_Apple_C"

        # 生成唯一名称
        unique_name = f"Apple_{int(time.time())}"

        # 使用 UnrealCV 命令生成苹果
        cmd = f"vset /objects/spawn_bp_asset {apple_bp} {unique_name}"
        res = self.communicator.unrealcv.client.request(cmd)

        print(f"Spawning apple: {res}")

        if "error" in res.lower() or "can not find" in res.lower():
            print(f"[ERROR] Failed to spawn apple: {res}")
            # 尝试备用路径
            apple_bp_alt = "/Game/SimWorld/Blueprints/Assets/BP_Apple_C.BP_Apple_C"
            cmd = f"vset /objects/spawn_bp_asset {apple_bp_alt} {unique_name}"
            res = self.communicator.unrealcv.client.request(cmd)
            print(f"Trying alternative path: {res}")

        # 设置苹果位置（在 agent 前方 5 单位）
        apple_loc = [loc[0] + 50, loc[1], loc[2] + 300]
        self.communicator.unrealcv.client.request(
            f"vset /object/{unique_name}/location {apple_loc[0]} {apple_loc[1]} {apple_loc[2] + 200}"
        )

        # 设置苹果缩放
        self.communicator.unrealcv.client.request(
            f"vset /object/{unique_name}/scale 200 200 200"
        )

        self.apple_location = apple_loc
        self.apple_name = unique_name
        self.phase = "SEEKING"

        print(f"Apple placed at {apple_loc} with name {unique_name}")

    def get_observation(self, role, video=""):
        """Get current observation from the environment."""
        agent = self.agents[role]
        loc_3d = self.communicator.unrealcv.get_location(self.agent_names[role])
        print(loc_3d)
        position = Vector(loc_3d[0], loc_3d[1])

        orientation = self.communicator.unrealcv.get_orientation(self.agent_names[role])
        yaw = orientation[1]
        direction = Vector(math.cos(math.radians(yaw)), math.sin(math.radians(yaw)))

        # Ensure agent.camera_id is bound to the humanoid's UE camera before reading
        try:
            if self.communicator is not None and self.agents[role] is not None:
                actor_name = self.communicator.get_humanoid_name(self.agents[role].id)
                found_cam = self.communicator.find_camera_id_for_actor(actor_name)
                if found_cam is not None:
                    self.agents[role].camera_id = found_cam
        except Exception:
            pass

        # Get camera observation (now should be the humanoid's view)
        try:
            ego_view = self.communicator.get_camera_observation(agent.camera_id, "lit")
        except Exception as e:
            print(f"[Warning] Camera read failed: {e}")
            ego_view = None

        return {
            "position": position,
            "direction": direction,
            "ego_view": ego_view,
            "video": video,
        }


def main():
    # 初始化环境和 agents
    env = HideAndSeekEnv(communicator, config)
    hider_agent = Agent(role="hider")
    seeker_agent = Agent(role="seeker", few_shot_data=few_shot_data)
    apple_placed = False
    done = False

    # Phase 1: Hider hides the apple
    print("\n=== PHASE 1: HIDER ===")
    obs = env.reset(role="hider")

    communicator.show_img(obs["ego_view"])

    max_hider_steps = 7
    for i in range(max_hider_steps):
        if obs["ego_view"] is None:
            print("Hider cannot see!")
            break

        print(f"\nHider Step {i}: Thinking...")
        action = hider_agent.action(
            obs, step=max_hider_steps - i, target=env.target, apple_placed=apple_placed
        )
        print(f"Hider Decision: {action}")

        communicator.sync_camera_to_actor(
            env.agents["hider"].id, env.agents["hider"].camera_id, height_offset=120
        )

        obs, reward, done, success, apple_placed = env.step(
            action, apple_placed=apple_placed, done=done, role="hider"
        )

    if not done:
        env.place_apple_at_agent()

        time.sleep(2)

    # Phase 2: Seeker finds the apple
    print("\n=== PHASE 2: SEEKER ===")
    obs = env.reset(role="seeker")
    done = False

    max_seeker_steps = 25
    for i in range(max_seeker_steps):
        if obs["ego_view"] is None:
            print("Seeker cannot see!")
            break

        print(f"\nSeeker Step {i}: Thinking...")
        action = seeker_agent.action(obs, step=max_seeker_steps - i)
        print(f"Seeker Decision: {action}")

        communicator.sync_camera_to_actor(
            env.agents["seeker"].id, env.agents["seeker"].camera_id, height_offset=120
        )

        obs, reward, done, success, apple_placed = env.step(
            action, done=done, role="seeker"
        )

        if done:
            break

        time.sleep(2)

    # 1. 安全创建根目录 (推荐使用 os.makedirs)
    base_records_dir = "./records"
    os.makedirs(base_records_dir, exist_ok=True)

    # 2. 创建本次运行的子文件夹 (建议使用整数时间戳或格式化日期)
    folder_name = str(int(time.time()))
    current_record_path = os.path.join(base_records_dir, folder_name)
    os.makedirs(current_record_path, exist_ok=True)

    # 3. 准备数据
    data = {
        "location": (obs["position"].x, obs["position"].y),
        "apple_placed": apple_placed,
        "sentence": sentence,
    }

    # 4. 写入 JSON (使用正确的路径变量)
    json_path = os.path.join(current_record_path, "data.json")
    with open(json_path, mode="w") as file:
        json.dump(data, file, indent=4)

    # 5. 保存图片 (必须传入路径)
    if "ego_view" in obs:
        last = Image.fromarray(cv2.cvtColor(obs["ego_view"], cv2.COLOR_BGR2RGB))
        img_path = os.path.join(current_record_path, "ego_view.png")
        last.save(img_path)
        print(f"Success: Saved data and image to {current_record_path}")


if __name__ == "__main__":
    main()
    print("\n=== GAME OVER ===")
    os._exit(0)
