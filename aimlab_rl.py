import cv2
import numpy as np
import pytesseract
from mss import mss
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
from gymnasium import spaces
import torch
import re
import interception
from interception import beziercurve
import pydirectinput
import time
from scipy import interpolate

#simualte human mouse movement 
def move_mouse_spline(start, end, duration=0.05, steps=30):
    mid_x = (start[0] + end[0]) / 2 + np.random.uniform(-10, 10)
    mid_y = (start[1] + end[1]) / 2 + np.random.uniform(-10, 10)
    x = [start[0], mid_x, end[0]]
    y = [start[1], mid_y, end[1]]
    
    tck, u = interpolate.splprep([x, y], k=2)
    u_fine = np.linspace(0, 1, steps)
    curve = interpolate.splev(u_fine, tck)
    points = list(zip(curve[0], curve[1]))
    deltas = [(int(points[i+1][0] - points[i][0]), int(points[i+1][1] - points[i][1]))
              for i in range(len(points) - 1)]
    for dx, dy in deltas:
        interception.move_relative(dx, dy)
        time.sleep(duration / steps)

#reading accuracy and score 
def clean_ocr_text(text):
    text = re.sub(r'[Oo]', '0', text)  
    text = re.sub(r'\D', '', text)         
    text = text.strip()                  
    return text

class AimLabEnv(gym.Env):
    def __init__(self):
        super(AimLabEnv, self).__init__()
        self.last_score = 0.0
        self.last_accuracy = 100.0

        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "screen": spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8),
            "bots": spaces.Box(low=0, high=1, shape=(10, 4), dtype=np.float32), 
            "bot_heads": spaces.Box(low=0, high=1, shape=(10, 4), dtype=np.float32), 
            "score": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "accuracy": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32)
        })
        
        self.yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp/weights/best.pt')
        self.yolo.to('cuda')
        self.sct = mss()
        self.monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}
        self.prev_score = 0

    def _get_obs(self):
        full_frame = np.array(self.sct.grab(self.monitor))[:, :, :3]
        height, width = full_frame.shape[:2]
        
        #trial and error to get correct roi for accuracy and score
        gray_full = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
        score_roi = gray_full[int(0.055*height):int(0.1*height), int(0.31*width):int(0.425*width)]
        accuracy_roi = gray_full[int(0.055*height):int(0.1*height), int(0.56*width):int(0.65*width)]

        '''
        cv2.imwrite("debug_score_roi.png", score_roi)
        cv2.imwrite("debug_accuracy_roi.png", accuracy_roi)
        
        debug_frame = full_frame.copy()

        # Draw rectangle around score region (BGR color format)
        cv2.rectangle(debug_frame, 
                    (int(0.31*width), int(0.055*height)),  # top-left corner
                    (int(0.425*width), int(0.1*height)),    # bottom-right corner
                    (0, 255, 0),  # Green color
                    2)  # Line thickness


        # Draw rectangle around accuracy region
        cv2.rectangle(debug_frame,
                    (int(0.56*width), int(0.055*height)),  # top-left corner
                    (int(0.65*width), int(0.1*height)),     # bottom-right corner
                    (0, 0, 255),  # Red color
                    2)  # Line thickness

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_frame, "Score", 
                    (int(0.31*width), int(0.05*height)), 
                    font, 0.5, (0, 255, 0), 1)
        cv2.putText(debug_frame, "Accuracy", 
                    (int(0.56*width), int(0.05*height)), 
                    font, 0.5, (0, 0, 255), 1)

        # Save debug image
        cv2.imwrite("debug_regions.png", debug_frame)
        print("[DEBUG] Saved ROI visualization to debug_regions.png")
        '''

        score_text = pytesseract.image_to_string(score_roi, config='--psm 7')
        score_text = clean_ocr_text(score_text)
        try:
            score = float(score_text)
            self.last_score = score
        except ValueError:
            score = self.last_score
            print(f"[WARN] OCR failed for score, using previous: {score}")

        accuracy_text = pytesseract.image_to_string(accuracy_roi, config='--psm 7') 
        accuracy_text = clean_ocr_text(accuracy_text)
        try:
            accuracy = float(accuracy_text)
            self.last_accuracy = accuracy
        except ValueError:
            accuracy = self.last_accuracy
            print(f"[WARN] OCR failed for accuracy, using previous: {accuracy}")

        results = self.yolo(full_frame)
        
        bots = np.zeros((10, 4), dtype=np.float32)
        bot_heads = np.zeros((10, 4), dtype=np.float32)
        bot_count, head_count = 0, 0

        for det in results.xyxy[0]:
            det = det.cpu().numpy().astype(np.float32)
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 1 and head_count < 10:  # Class 0: bot head
                bot_heads[head_count] = [x1 / width, y1 / height, x2 / width, y2 / height]
                head_count += 1
                print(f"[DEBUG] Bot head detected: {bot_heads[head_count-1]} (norm coords)")
            elif int(cls) == 0 and bot_count < 10: # Class 1: bot body
                bots[bot_count] = [x1 / width, y1 / height, x2 / width, y2 / height]
                bot_count += 1
                print(f"[DEBUG] Bot detected: {bots[bot_count-1]} (norm coords)")

        print(f"[DEBUG] Total bot heads detected: {head_count}")
        print(f"[DEBUG] Total bots detected: {bot_count}")

        small_frame = cv2.resize(full_frame, (84, 84))
        

        if len(results.xyxy[0]) > 0:
            debug_frame = small_frame.copy()
            for i in range(head_count):
                x1 = int(bot_heads[i][0] * 84)
                y1 = int(bot_heads[i][1] * 84)
                x2 = int(bot_heads[i][2] * 84)
                y2 = int(bot_heads[i][3] * 84)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(debug_frame, 'Head', (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

            for i in range(bot_count):
                x1 = int(bots[i][0] * 84)
                y1 = int(bots[i][1] * 84)
                x2 = int(bots[i][2] * 84)
                y2 = int(bots[i][3] * 84)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(debug_frame, 'Bot', (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)

            cv2.imwrite("debug_agent_view.png", debug_frame)
            print("[DEBUG] Saved annotated agent view to debug_agent_view.png")
        
        return {
            "screen": small_frame,
            "bots": bots,
            "bot_heads": bot_heads,
            "score": np.array([score], dtype=np.float32),
            "accuracy": np.array([accuracy], dtype=np.float32)
        }

    def step(self, action):
        print("action", action)
        dx, dy, shoot, w, a, s, d = action
        '''
        x, y = interception.mouse_position()
        new_x = x + int(dx * 10)
        new_y = y + int(dy * 10)
        interception.move_to(new_x, new_y)
        '''
        start_x, start_y = interception.mouse_position()
        end_x = start_x + int(dx * 84)
        end_y = start_y + int(dy * 84)
        move_mouse_spline((start_x, start_y), (end_x, end_y), duration=0.05, steps=20)

        if shoot > 0.5:
            interception.click(button="left")

        movement_keys = {
            'w': w > 0.5,
            'a': a > 0.5,
            's': s > 0.5,
            'd': d > 0.5
        }
        for key, is_pressed in movement_keys.items():
            if is_pressed:
                pydirectinput.keyDown(key)
            else:
                pydirectinput.keyUp(key)

        obs = self._get_obs()
        reward = obs["score"][0] - self.prev_score
        self.prev_score = obs["score"][0]
        terminated = bool(obs["accuracy"][0] < 50)
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prev_score = 0
        observation = self._get_obs()
        for key in ['w', 'a', 's', 'd']:
            pydirectinput.keyUp(key)
        info = {}
        return observation, info

interception.auto_capture_devices()
params = interception.beziercurve.BezierCurveParams()
params.duration = 0.05
params.smoothness = 0.002
beziercurve.set_default_params(params)
env = AimLabEnv()
check_env(env)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
model.save("aimlab_ppo")