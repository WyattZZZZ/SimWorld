"""Utilities for recording simulation videos and logging metadata."""

import csv
import os
import random
import threading
import time

import cv2


class VideoRecorder:
    """Record frames from a humanoid-mounted camera and export video/CSV logs."""

    def __init__(
            self,
            communicator,
            humanoid,
            output_dir='.',
            resolution=(1440, 720),
            fps=25.0,
            frame_num=500,
            move_pattern=None,
            camera_mode='lit'):
        """Initialize the recorder with target communicator and humanoid.

        Parameters
        ----------
        communicator: object
            Game communicator providing camera and control APIs.
        humanoid: object
            Humanoid agent that owns the camera to record from.
        output_dir: str
            Directory to write the output files (video and CSV logs).
        video_name: str
            Output video filename.
        resolution: tuple[int, int]
            Target camera resolution (width, height).
        fps: float
            Frames per second for the output video.
        frame_num: int
            Total number of frames to record. Use >0 to enable progress.
        camera_mode: str
            Unreal render mode, e.g., 'lit', 'depth', etc.
        """
        self.communicator = communicator
        self.humanoid = humanoid
        self.output_dir = output_dir
        self.video_path = os.path.join(output_dir, f'{self.humanoid.id} - {time.time()}.mp4')
        self.csv_camera = os.path.join(output_dir, 'camera_position.csv')
        self.csv_action = os.path.join(output_dir, 'humanoid_action.csv')
        self.resolution = resolution
        self.fps = fps
        self.frame_num = frame_num
        self.camera_mode = camera_mode
        if move_pattern:
            self.move_pattern = lambda timestamp: move_pattern(self, timestamp)
        else:
            self.move_pattern = self._move_pattern

        # set resolution
        self.communicator.unrealcv.set_camera_resolution(
            self.humanoid.camera_id, resolution
        )

        os.makedirs(output_dir, exist_ok=True)

        # Async recording state
        self.is_recording = False
        self._recording_thread = None
        self._frames = []
        self._camera_positions = []
        self._actions = []
        self._start_time = None

    def record(self):
        """Run the recording loop synchronously and persist video and CSV logs.

        Returns
        -------
        str
            Path to the saved video file.
        """
        self._frames = []
        self._camera_positions = []
        self._actions = []
        self.is_recording = True

        timestamp = 0
        while len(self._frames) < self.frame_num:
            # Sync camera and get pose in one step
            cam_loc, cam_rot = self.communicator.sync_camera_to_actor(self.humanoid.id, self.humanoid.camera_id)
            self._capture_step(timestamp, cam_loc, cam_rot)

            # Tick logic for synchronous recording
            self.communicator.unrealcv.tick()
            timestamp += 1

            # Progress bar
            progress = len(self._frames) / self.frame_num
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '#' * filled + '-' * (bar_length - filled)
            print(f'\rRecording: |{bar}| {len(self._frames)}/{self.frame_num} ({progress:.1%})', end='')

        self.is_recording = False
        print('\n✅ Sync recording finished.')
        self.save_video(self._frames)
        self.save_csv(self._camera_positions, self._actions)
        return self.video_path

    def _capture_step(self, timestamp, camera_loc=None, camera_rot=None):
        """Capture a single frame and log metadata.

        Args:
            timestamp: Step timestamp.
            camera_loc: Optional pre-calculated camera location.
            camera_rot: Optional pre-calculated camera rotation.
        """
        try:
            # If pose not provided, sync and get it now
            if camera_loc is None or camera_rot is None:
                camera_loc, camera_rot = self.communicator.sync_camera_to_actor(
                    self.humanoid.id, self.humanoid.camera_id
                )

            img = self.communicator.get_camera_observation(
                self.humanoid.camera_id, self.camera_mode, mode='direct'
            )
            
            # Use cached pose instead of querying UE again
            self._frames.append(img)
            self._camera_positions.append([timestamp, camera_loc, camera_rot])

            # Record move pattern if needed (only for internal pattern)
            # In async mode, actions are usually driven by external logic
            if not self._recording_thread:
                actions = self.move_pattern(timestamp)
                self._actions.extend(actions)
                
        except Exception as e:
            print(f'\n❌ Error during capture at step {timestamp}: {e}')

    def start_async(self):
        """Start recording in a background thread."""
        if self.is_recording:
            print("⚠️ Already recording.")
            return

        self._frames = []
        self._camera_positions = []
        self._actions = []
        self.is_recording = True
        self._start_time = time.time()

        self._recording_thread = threading.Thread(target=self._async_recording_loop, daemon=True)
        self._recording_thread.start()
        print(f"🎬 Async recording started (FPS: {self.fps}).")

    def stop_async(self, preserve_real_time=True):
        """Stop background recording and save files.

        Args:
            preserve_real_time (bool): If True, adjust video FPS to match actual capture rate 
                                      so the video duration matches real-world duration.
        """
        if not self.is_recording:
            print("⚠️ Not recording.")
            return

        self.video_path = os.path.join(self.output_dir, f'{self.humanoid.id} - {time.time()}.mp4')
        self.is_recording = False
        duration = time.time() - self._start_time
        if self._recording_thread:
            self._recording_thread.join()
        if len(self._frames) <= 1:
            print('\n❌ No frames captured. Recording failed.')
            return None
        
        actual_fps = len(self._frames) / duration if duration > 0 else self.fps
        print('\n✅ Async recording finished.')
        print(f'📊 Stats: Captured {len(self._frames)} frames in {duration:.2f}s (Actual FPS: {actual_fps:.2f}).')
        save_fps = actual_fps if preserve_real_time else self.fps
        if preserve_real_time:
            print(f'🎬 Saving video at {save_fps:.2f} FPS to match simulation duration.')
            
        self.save_video(self._frames, override_fps=save_fps)
        self.save_csv(self._camera_positions, self._actions)
        return self.video_path

    def _async_recording_loop(self):
        """Background loop for capturing frames."""
        interval = 1.0 / self.fps
        timestamp = 0
        
        while self.is_recording:
            loop_start = time.time()
            
            self._capture_step(timestamp)
            timestamp += 1
            
            # Sleep to maintain FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _move_pattern(self, timestamp):
        """Generate a movement pattern for the humanoid at given timestamp."""
        actions = []
        # In async mode, we don't automatically trigger move patterns
        # because the user is controlling the agent externally.
        if self._recording_thread:
            return actions

        step = timestamp
        # --- move forward ---
        self._apply_action('move_forward', None, timestamp, actions)

        # --- big turn ---
        if step % random.randint(30, 50) == 0:
            turn_angle = random.randint(60, 160)
            turn_dir = random.choice(['left', 'right'])
            action = f'rotate_{turn_dir}'
            self._apply_action(action, turn_angle, timestamp, actions)

        # --- small jitter ---
        elif step % random.randint(20, 40) == 0:
            jitter_angle = random.randint(-10, 10)
            if jitter_angle > 0:
                self._apply_action('rotate_right', jitter_angle, timestamp, actions)
            elif jitter_angle < 0:
                self._apply_action('rotate_left', abs(jitter_angle), timestamp, actions)

        return actions

    def _apply_action(self, action, value, timestamp, actions):
        """Apply a single action and log it into the actions list."""
        if action == 'move_forward':
            self.communicator.humanoid_move_forward(self.humanoid.id)
            actions.append([timestamp, 'move_forward'])
        elif action == 'rotate_right':
            self.communicator.humanoid_rotate(self.humanoid.id, value, 'right')
            actions.append([timestamp, 'rotate_right'])
        elif action == 'rotate_left':
            self.communicator.humanoid_rotate(self.humanoid.id, value, 'left')
            actions.append([timestamp, 'rotate_left'])

    def save_video(self, images, override_fps=None):
        """Save frames to an MP4 file using OpenCV."""
        if not images:
            print('⚠️ No frames to save.')
            return
            
        fps = override_fps if override_fps is not None else self.fps
        h, w, _ = images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_path, fourcc, fps, (w, h))
        for img in images:
            out.write(img)  # img must be in BGR format
        out.release()
        print(f'💾 Video saved to {self.video_path} at {fps:.2f} FPS')

    def save_csv(self, camera_position, humanoid_actions):
        """Save camera pose timeline and humanoid actions into CSV files."""
        with open(self.csv_camera, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'camera_loc', 'camera_rot'])
            for loc in camera_position:
                writer.writerow(loc)
        print(f'💾 Camera positions saved to {self.csv_camera}')

        with open(self.csv_action, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'action'])
            for action in humanoid_actions:
                writer.writerow(action)
        print(f'💾 Actions saved to {self.csv_action}')
