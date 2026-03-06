import os
import json
import time
import shutil
from PIL import Image


class DataRecorder:
    def __init__(self, round_id: int):
        """
        初始化记录器
        :param round_id: 传入当前进程的 round 编号，防止并行冲突
        """
        self.root_data_dir = os.path.abspath("dataset")
        # 创建一个唯一的实验运行文件夹：时间戳_roundID
        self.session_id = f"{int(time.time())}_round_{round_id}"
        self.session_path = os.path.join(self.root_data_dir, self.session_id)

        if not os.path.exists(self.session_path):
            os.makedirs(self.session_path, exist_ok=True)

        print(f"[Recorder] Data will be saved to: {self.session_path}")

    def record_step(self, role: str, step_num: int, obs: dict, action: str):
        """
        记录一步的所有数据
        :param role: 'hider' 或 'seeker'
        :param step_num: 当前步数序号
        :param obs: 包含 position, direction, ego_view, video 的字典
        :param action: 动作字符串
        """
        # 1. 创建步骤文件夹，例如: data/session_id/hider/step_1
        step_dir = os.path.join(self.session_path, role, f"step_{step_num}")
        os.makedirs(step_dir, exist_ok=True)

        # 2. 处理并保存 JSON 数据 (obs 中的 position 和 direction 需要转为可序列化格式)
        json_data = {
            "action": action,
            "obs": {
                "position": {
                    "x": getattr(obs["position"], "x", obs["position"]),
                    "y": getattr(obs["position"], "y", obs["position"]),
                },
                "direction": {
                    "x": getattr(obs["direction"], "x", obs["direction"]),
                    "y": getattr(obs["direction"], "y", obs["direction"]),
                },
            },
        }

        with open(os.path.join(step_dir, "data.json"), "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

        # 3. 保存图片 (ego_view)
        if obs.get("ego_view") is not None:
            try:
                img = Image.fromarray(obs["ego_view"])
                img.save(os.path.join(step_dir, "view.png"))
            except Exception as e:
                print(f"[Recorder Error] Failed to save image: {e}")

        # 4. 保存视频 (video)
        # 假设 obs["video"] 提供的是视频文件的临时路径，我们将其移动/复制到目标文件夹
        temp_video_path = obs.get("video")
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                # 使用 copy 避免因为原始文件被占用导致删除失败
                shutil.copy(temp_video_path, os.path.join(step_dir, "video.mp4"))
            except Exception as e:
                print(f"[Recorder Error] Failed to save video: {e}")
