import base64
import io
import json
import os
import re
import time

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel
from google.genai import types

from simworld.utils.logger import Logger

from .base_llm import BaseLLM


class A2ALLM(BaseLLM):
    """Local Planner (Activity to Action) LLM class for handling interactions with language models."""
    def __init__(self, model_name: str = 'gpt-4o-mini', url: str = None, provider: str = 'openai'):
        """Initialize the Local Planner LLM."""
        super().__init__(model_name, url, provider)

        self.logger = Logger.get_logger('A2ALLM')

    def generate_instructions(self, system_prompt, user_prompt, images=[], video_path=None, max_tokens=None, temperature=0.7, top_p=1.0, response_format=BaseModel):
        """Generate instructions for the Local Planner system.

        Args:
            system_prompt (str): The system prompt for the Local Planner system.
            user_prompt (str): The user prompt for the Local Planner system.
            images (list): The images for the Local Planner system.
            video_path (str): Optional path to a video file.
            max_tokens (int): The maximum number of tokens for the Local Planner system.
            temperature (float): The temperature for the Local Planner system.
            top_p (float): The top_p for the Local Planner system.
            response_format (BaseModel): The response format for the Local Planner system.
        """
        if self.provider == 'openai':
            return self._generate_instructions_openai(system_prompt, user_prompt, images, video_path, max_tokens, temperature, top_p, response_format)
        elif self.provider == 'openrouter':
            return self._generate_instructions_openrouter(system_prompt, user_prompt, images, video_path, max_tokens, temperature, top_p, response_format)
        elif self.provider == 'google':
            return self._generate_instructions_gemini(system_prompt, user_prompt, images, video_path, max_tokens, temperature, top_p, response_format)
        else:
            raise ValueError(f'Invalid provider: {self.provider}')

    def _generate_instructions_openai(self, system_prompt, user_prompt, images=[], video_path=None, max_tokens=None, temperature=0.7, top_p=1.0, response_format=BaseModel):
        start_time = time.time()
        user_content = []
        user_content.append({'type': 'text', 'text': user_prompt})

        for image in images:
            img_data = self._process_image_to_base64(image)
            user_content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/jpeg;base64,{img_data}'}
            })

        if video_path:
            video_data = self._process_video_to_base64(video_path)
            # Standard video format for many VLMs (like Qwen-VL)
            user_content.append({
                'type': 'video_url',
                'video_url': {'url': f'data:video/mp4;base64,{video_data}'}
            })
        
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            action_json = response.choices[0].message.content
            if "Think" in self.model_name or "think" in self.model_name:
                valid_commands = ['forward', 'rotate', 'place_apple', 'found', 'wait']
                lines = action_json.split('\n')
                for line in reversed(lines):  # 从后往前找，通常指令在最后
                    line = line.strip().lower()
                    if any(cmd in line for cmd in valid_commands):
                        action_json = line
                        break
        except Exception as e:
            self.logger.error(f'Error in generate_instructions_openai: {e}')
            action_json = None

        return action_json, time.time() - start_time

    def _generate_instructions_gemini(self, system_prompt, user_prompt, images=[], video_path=None, max_tokens=None, temperature=0.7, top_p=1.0, response_format=BaseModel):
        start_time = time.time()
        
        parts = []
        # Add system prompt and user prompt
        parts.append(types.Part(text=system_prompt + '\n' + user_prompt))
        
        # Add images
        for image in images:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()
            parts.append(types.Part(inline_data=types.Blob(data=img_bytes, mime_type='image/jpeg')))

        # Add video if provided (User recommended direct bytes for videos < 20MB)
        if video_path:
            self.logger.info(f"Adding video direct bytes to Gemini: {video_path}")
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            parts.append(types.Part(inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')))

        # Create the prompt with text and multiple images/videos
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=types.Content(parts=parts)
        )

        action_json = response.text

        return action_json, time.time() - start_time

    def _generate_instructions_openrouter(self, system_prompt, user_prompt, images=[], video_path=None, max_tokens=None, temperature=0.7, top_p=1.0, response_format=BaseModel):

        start_time = time.time()
        user_content = []
        user_prompt += '\nPlease respond in valid JSON format following this schema: ' + str(response_format.to_json_schema())
        user_content.append({'type': 'text', 'text': user_prompt})

        for image in images:
            img_data = self._process_image_to_base64(image)
            user_content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/jpeg;base64,{img_data}'}
            })

        if video_path:
            video_data = self._process_video_to_base64(video_path)
            user_content.append({
                'type': 'video_url',
                'video_url': {'url': f'data:video/mp4;base64,{video_data}'}
            })

        action_response = None
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_content}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            action_response = response.choices[0].message.content
        except Exception as e:
            self.logger.error(f'Error in generate_instructions_openrouter: {e}')
            action_response = None

        if action_response is None:
            self.logger.warning('Warning: Failed to get action response, using default')
            action_json = None
        else:
            action_json = self._extract_json_and_fix_escapes(action_response)

        return action_json, time.time() - start_time

    def _process_image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy array image to base64 string.

        Args:
            image (np.ndarray): Image array (1 or 3 channels)

        Returns:
            str: Base64 encoded image string
        """
        # Convert single channel to 3 channels if needed
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Ensure uint8 type
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Convert to PIL Image
        pil_image = Image.fromarray(image)

        # Convert to base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format='JPEG')
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str

    def _process_video_to_base64(self, video_path: str) -> str:
        """Read a video file and encode it to base64.

        Args:
            video_path (str): Path to the video file.

        Returns:
            str: Base64 encoded video string.
        """
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return ""

        with open(video_path, "rb") as video_file:
            return base64.b64encode(video_file.read()).decode('utf-8')

    def _extract_json_and_fix_escapes(self, text):
        # Extract content from first { to last }
        pattern = r'(\{.*\})'
        match = re.search(pattern, text, re.DOTALL)

        if match:
            json_str = match.group(1)
            # Fix invalid escape sequences in JSON
            fixed_json = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
            try:
                # Try to parse the fixed JSON
                json_obj = json.loads(fixed_json)
                return json_obj
            except json.JSONDecodeError as e:
                self.logger.error(f'JSON parsing error: {e}')
                self.logger.error(f'Fixed JSON: {fixed_json}')
                # Return the fixed string if parsing fails
                return fixed_json
        else:
            self.logger.error(f'No JSON found in the text: {text}')
            return None
