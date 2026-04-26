import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics, HumanRendering
import pygame
import os

# Initialize with 'rgb_array' for recording
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Add Recording Wrappers (Note: order matters)
# Records every episode automatically
env = RecordVideo(env, video_folder="recorded_videos", episode_trigger=lambda x: True)
env = RecordEpisodeStatistics(env)

# Layer HumanRendering so game is visible while playing
env = HumanRendering(env)

# function to get player input
def get_action():
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] or keys[pygame.K_UP]: return 2
    if keys[pygame.K_a] or keys[pygame.K_LEFT]: return 1
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]: return 3
    return 0

# Reset and Loop
pygame.init()
clock = pygame.time.Clock()

running = True
while running:
    observation, info = env.reset()

    terminated = False
    truncated = False

    print("Game Started! Use WASD or Arrows.")

    while not (terminated or truncated):
        # Keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                terminated = True
                break

        action = get_action()
        
        # Step through the environment
        observation, reward, terminated, truncated, info = env.step(action)
        
        # Cap the speed so the game is playable (60 FPS)
        clock.tick(60)

        # when user presses close button, exit outer loop
        if pygame.event.get() == pygame.QUIT:
            running = False

    # Stats extraction from NumPy arrays
    if "episode" in info:
        # float() and int() handle the new NumPy array return types
        print(f"\n--- Episode Stats ---")
        print(f"Total Reward: {float(info['episode']['r']):.2f}")
        print(f"Steps: {int(info['episode']['l'])}")

# env.close() finalizes the video file
env.close()
pygame.quit()
