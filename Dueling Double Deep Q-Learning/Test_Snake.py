from Environment import *
import torch

env = SnakeEnv()
EPISODES = 500
for episode in range(EPISODES):
    score = 0
    done = False
    image = env.reset(stop_render = False)
    env.render()
    action = 0
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    action = 3
                elif event.key == pygame.K_LEFT:
                    action = 2
                elif event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_DOWN:
                    action = 1
        new_state, reward, done = env.step(action)
        score += reward
    print("episode : {} | score : {} ".format(episode, score))
