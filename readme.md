# Project Overview
This project presents a novel approach to training an AI agent to play a first-person shooter (FPS) aim trainer, specifically Aim Lab. The core of this work is an AI agent that learns to aim and shoot at targets in real-time by interacting directly with the game screen.

The project leverages a custom Gymnasium environment that wraps the Aim Lab game, allowing a Reinforcement Learning (RL) model to observe the game state and perform actions (moving the mouse, shooting, and character movement). 

## Motivation
The primary goal of this project is to explore the capabilities of Reinforcement Learning in a complex, real-world control problem. By creating an environment that simulates human-like interaction with a game, I believe that the trained agent can master a skill—accurate aiming. Bonus points if I am able to learn tips and tricks from the model

## How It Works
The project is structured around a custom Gym environment, AimLabEnv, which serves as the interface between the game and the reinforcement learning model.

1. Environment Setup (AimLabEnv)
The AimLabEnv class defines the core logic for the RL loop:

Observation Space: The agent's perception of the game world is defined by a multi-modal observation space. This includes:

screen: A downscaled 84x84 pixel image of the game screen, providing visual context.

bots & bot_heads: Bounding box coordinates (normalized) for detected bot bodies and heads. This is the primary target information for the agent.

score & accuracy: Real-time score and accuracy metrics extracted from the game UI.

Action Space: The agent controls the game through a 7-dimensional continuous action space, representing:

dx, dy: Relative mouse movement along the x and y axes.

shoot: A value between -1 and 1, where a value greater than a threshold (e.g., 0.5) triggers a mouse click.

w, a, s, d: Binary-like controls for character movement (forward, left, backward, right).

Object Detection: A pre-trained YOLOv5 model (yolov5/runs/train/exp/weights/best.pt) is used to detect the location of bots and bot heads on the screen. The bounding boxes are converted to a structured NumPy array for the agent to process.

OCR: The pytesseract library is used to perform OCR on specific regions of interest (ROIs) on the screen to read the current score and accuracy which acts as the rewards punishment.

2. Human-like Mouse Movement
To make the agent's aiming feel more natural and less prone to detection in online scenarios, a custom function move_mouse_spline is used. Instead of a direct, instantaneous jump to a new coordinate, this function:

Generates a smooth, curved path using a cubic spline interpolation between the starting point, a randomized midpoint, and the target end point.

The mouse is moved incrementally along this path, simulating the nuanced, non-linear movement of a human hand.

3. Reinforcement Learning Loop (step and reset)
step(action): This function takes the agent's action and translates it into in-game commands.

Mouse movement is handled by the move_mouse_spline function.

Shooting is triggered by simulating a mouse click using interception.

Character movement is controlled via pydirectinput.keyDown and pydirectinput.keyUp.

After performing the action, it calls _get_obs() to get the new state.

Reward: The reward is calculated as the change in score from the previous step. This simple but effective reward signal encourages the agent to hit targets and increase its score.

Termination: The episode terminates if the agent's accuracy drops below a certain threshold (e.g., 50%), forcing it to learn to be precise.

reset(): This function prepares the environment for a new episode. It resets the internal state, ensures all movement keys are released, and returns the initial observation.

4. Training
The project uses the Stable-Baselines3 library, a robust framework for training RL agents.

Policy: A PPO (Proximal Policy Optimization) model with a MultiInputPolicy is used. This policy is designed to handle the multi-modal dictionary observation space, effectively processing both screen images and structured data (bounding boxes, scores).

Training Loop: The model is trained for a total of 10,000 timesteps, and the trained model is saved for future use.
