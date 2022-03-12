import pygame
import random
import os

ROOT_DIR = os.path.dirname(__file__)

pygame.init()
red = (255, 0, 0)
red1 = (255, 51, 51)
orange = (255, 128, 0)
orange1 = (255, 153, 51)
yellow = (255, 255, 0)
green = (128, 255, 0)
green1 = (0, 255, 0)
green2 = (0, 255, 128)
blue = (0, 255, 255)
dblue = (0, 128, 255)
dblue1 = (0, 0, 255)
violet = (127, 0, 255)
pink = (255, 0, 255)
dpink = (255, 0, 127)
white = (255, 255, 255)
black = (0, 0, 0)


class Snake:
    """
    Snake class representing the snake
    """

    def __init__(self):
        """
        Initialize the object
        """
        self.snakeX_change = 0
        self.snakeY_change = 0
        self.snakeX = 300
        self.snakeY = 325
        self.head = []
        self.snake_list = []
        self.length_snake = 1

    def draw(self, screen):
        screen.fill((0, 0, 0))
        for x, y in self.snake_list:
            pygame.draw.rect(screen, green, (x + 3, y + 3, 20, 20))

    def eating(self, foodX, foodY):
        if foodX == self.snakeX and foodY == self.snakeY:
            return True
        else:
            return False

    def move(self):
        self.snakeX += self.snakeX_change
        self.snakeY += self.snakeY_change
        self.head = [self.snakeX, self.snakeY]
        self.snake_list.append(self.head)

    def update(self):
        if len(self.snake_list) > self.length_snake:
            del (self.snake_list[0])

    def turn_right(self):
        self.snakeX_change = 0
        self.snakeY_change = 0
        self.snakeX_change = 25

    def turn_left(self):
        self.snakeX_change = 0
        self.snakeY_change = 0
        self.snakeX_change = -25

    def turn_up(self):
        self.snakeX_change = 0
        self.snakeY_change = 0
        self.snakeY_change = -25

    def turn_down(self):
        self.snakeX_change = 0
        self.snakeY_change = 0
        self.snakeY_change = 25

    def check_collision(self, reward):
        if self.snakeX < 0 or self.snakeX > 600 or self.snakeY < 50 or self.snakeY > 650:
            return -5, True

        for x, y in self.snake_list[:-1]:
            if [x, y] == self.head:
                return -5, True
        return reward, False


class Food:
    """
    represents a food object
    """

    def __init__(self):
        """
        initialize food object
        """
        self.foodX = round((random.randint(0, 600)) / 25.0) * 25
        self.foodY = round((random.randint(50, 650)) / 25.0) * 25

    def food(self, screen):
        pygame.draw.rect(screen, red, (self.foodX + 3, self.foodY + 3, 20, 20))

    def generate_food(self):
        self.foodX = round((random.randint(0, 600)) / 25.0) * 25
        self.foodY = round((random.randint(50, 650)) / 25.0) * 25

    def draw(self, screen):
        pygame.draw.rect(screen, red, (self.foodX + 3, self.foodY + 3, 20, 20))


def message_to_screen(screen, q, size, colour, x, y):
    message = pygame.font.Font('freesansbold.ttf', size)
    messages = message.render(q, True, colour)
    screen.blit(messages, (x, y))


class SnakeEnv(object):
    pygame.display.init()
    win = pygame.display.set_mode((625, 675), flags = pygame.HIDDEN)
    pygame.display.set_caption("SNAKE")
    icon = pygame.image.load(os.path.realpath(os.path.join(ROOT_DIR, "assets/cobra.png")))
    pygame.display.set_icon(icon)
    logo = pygame.image.load(os.path.realpath(os.path.join(ROOT_DIR, 'assets/scared.png')))

    def __init__(self):
        self.WIN_WIDTH = 625
        self.WIN_HEIGHT = 675
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        self.clock = pygame.time.Clock()
        self.lost = False
        self.action_space = [0, 1, 2, 3]
        self.observation_space = (self.WIN_WIDTH, self.WIN_HEIGHT, 3)

    def render(self):
        pygame.display.init()
        SnakeEnv.win = pygame.display.set_mode((self.WIN_WIDTH, self.WIN_HEIGHT))
        pygame.display.set_caption("SNAKE")
        pygame.display.set_icon(self.icon)

    def Stop_render(self):
        pygame.display.init()
        SnakeEnv.win = pygame.display.set_mode((self.WIN_WIDTH, self.WIN_HEIGHT), flags = pygame.HIDDEN)
        pygame.display.set_caption("SNAKE")
        pygame.display.set_icon(self.icon)

    def step(self, action):
        pygame.event.pump()
        self.clock.tick(10)
        reward = 0.1
        if action == 0:
            self.snake.turn_up()
        elif action == 1:
            self.snake.turn_down()
        elif action == 2:
            self.snake.turn_left()
        else:
            self.snake.turn_right()
        self.snake.move()
        if self.snake.eating(self.food.foodX, self.food.foodY):
            self.score += 1
            reward = 2
            self.snake.length_snake += 1
            self.food.generate_food()

        self.snake.update()
        reward, self.lost = self.snake.check_collision(reward)
        image = self.draw_window()
        return image, reward, self.lost

    def reset(self, stop_render=True):
        self.snake = Snake()
        self.food = Food()
        self.score = 0
        if stop_render:
            self.Stop_render()
        self.clock = pygame.time.Clock()
        self.lost = False
        image = self.draw_window()
        return image

    def draw_window(self):
        """
        draws the windows for the main game loop
        :return: image
        """
        self.snake.draw(self.win)
        self.food.draw(self.win)
        for i in range(0, 650, 25):
            for j in range(0, 700, 25):
                if j >= 50:
                    pygame.draw.line(self.win, (255, 255, 255), (0, j), (625, j), 1)
                pygame.draw.line(self.win, (255, 255, 255), (i, 50), (i, 675), 1)

        pygame.draw.rect(self.win, dblue, (0, 0, 625, 50))
        pygame.draw.line(self.win, yellow, (0, 0), (625, 0), 10)
        pygame.draw.line(self.win, yellow, (0, 47), (625, 47), 5)

        message_to_screen(self.win, "Score : " + str(self.score), 25, (255, 255, 255), 15, 15)
        message_to_screen(self.win, "SNAKE  XENZIA", 40, white, 225, 10)
        self.win.blit(self.logo, (180, 12))
        pygame.display.update()
        array = pygame.surfarray.array3d(self.win)
        return array
