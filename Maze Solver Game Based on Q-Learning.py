# 安装依赖（Colab运行必备）
!apt-get update > /dev/null 2>&1
!apt-get install -y xvfb > /dev/null 2>&1
!pip install pyvirtualdisplay pygame numpy pillow ipywidgets > /dev/null 2>&1

# 导入模块
from pyvirtualdisplay import Display
import pygame
import numpy as np
import random
import IPython.display
from PIL import Image
from ipywidgets import Button, HBox, Output, Layout

# 启动虚拟显示（适配Colab）
display = Display(visible=0, size=(1600, 900))
display.start()

# 游戏配置
class Config:
    MAZE_SIZE = 5          # 迷宫尺寸5x5
    WALL_DENSITY = 0.2     # 墙的生成概率
    CELL_SIZE = 120        # 单个格子像素大小
    SCREEN_WIDTH = 1600    # 窗口宽度
    SCREEN_HEIGHT = 900    # 窗口高度

    # Q-Learning参数
    LEARNING_RATE = 0.1
    DISCOUNT_FACTOR = 0.9
    INIT_EPSILON = 1.0
    FINAL_EPSILON = 0.05
    EPSILON_DECAY = 0.001
    MAX_STEPS = 100         # 每轮最大步数
    TOTAL_EPISODES = 1000   # 总训练轮数

    # 颜色配置
    TEXT_COLOR = (255, 255, 255)
    BG_COLOR = (0, 0, 0)
    BUTTON_BG = (50, 50, 50)
    BUTTON_BORDER = (200, 200, 200)

# 迷宫类：生成迷宫、判断墙
class Maze:
    def __init__(self, size, wall_density):
        self.size = size
        self.wall_density = wall_density
        self.grid = self._generate_maze()

    # 生成迷宫网格（0=空地，1=墙）
    def _generate_maze(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        for x in range(self.size):
            for y in range(self.size):
                # 起点/终点不生成墙
                if (x == 0 and y == 0) or (x == self.size-1 and y == self.size-1):
                    continue
                if random.random() < self.wall_density:
                    grid[x][y] = 1
        return grid

    # 判断坐标是否为墙（越界也算墙）
    def is_wall(self, x, y):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return True
        return self.grid[x][y] == 1

# Q-Learning智能体：决策、更新Q表
class QLearningAgent:
    def __init__(self, maze_size, lr, gamma):
        self.state_size = maze_size * maze_size  # 状态数=格子数
        self.action_size = 4                     # 动作：上下左右
        self.lr = lr
        self.gamma = gamma
        self.epsilon = Config.INIT_EPSILON       # 探索率
        self.q_table = np.zeros((self.state_size, self.action_size))  # Q表

    # 坐标转状态值
    def get_state(self, x, y):
        return x * Config.MAZE_SIZE + y

    # ε-贪心选动作
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size-1)  # 随机探索
        else:
            return np.argmax(self.q_table[state])         # 最优选择

    # 更新Q表
    def update_q_table(self, state, action, reward, next_state):
        max_next_q = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (reward + self.gamma * max_next_q - self.q_table[state][action])

    # 衰减探索率
    def decay_epsilon(self):
        self.epsilon = max(Config.FINAL_EPSILON, self.epsilon - Config.EPSILON_DECAY)

# 初始化游戏
pygame.init()
screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
pygame.display.set_caption("Maze Solver - Q-Learning")

# 字体配置
font_large = pygame.font.SysFont(None, 80)
font_medium = pygame.font.SysFont(None, 45)
font_small = pygame.font.SysFont(None, 30)

# 游戏状态
GAME_STATE_START = 0    # 开始界面
GAME_STATE_PLAYING = 1  # 游戏中
GAME_STATE_END = 2      # 一轮结束

# 全局变量初始化
maze = Maze(Config.MAZE_SIZE, Config.WALL_DENSITY)
agent = QLearningAgent(Config.MAZE_SIZE, Config.LEARNING_RATE, Config.DISCOUNT_FACTOR)
agent_x, agent_y = 0, 0  # 智能体初始位置
current_episode = 0      # 当前轮数
current_steps = 0        # 当前步数
cumulative_reward = 0    # 累计奖励
training = True          # 训练/游玩模式
game_state = GAME_STATE_START
episode_reward = 0

# 输出框配置
out_display = Output(layout=Layout(border='1px solid black', min_height='900px', min_width='1600px'))

# 绘制开始界面
def draw_start_screen():
    screen.fill(Config.BG_COLOR)

    # 标题
    title_text = font_large.render("Maze Q-Learning", True, Config.TEXT_COLOR)
    title_rect = title_text.get_rect(center=(Config.SCREEN_WIDTH//2, 100))
    screen.blit(title_text, title_rect)

    # 规则说明
    desc1 = font_medium.render("Game Rules:", True, Config.TEXT_COLOR)
    desc2 = font_small.render("1. Blue=Agent, Green=Start, Red=Goal", True, Config.TEXT_COLOR)
    desc3 = font_small.render("2. Training: Agent learns path", True, Config.TEXT_COLOR)
    desc4 = font_small.render("3. Play: Agent uses learned path", True, Config.TEXT_COLOR)
    desc5 = font_small.render("Click 'Start Game' to begin", True, Config.TEXT_COLOR)

    desc1_rect = desc1.get_rect(center=(Config.SCREEN_WIDTH//2, 220))
    desc2_rect = desc2.get_rect(center=(Config.SCREEN_WIDTH//2, 300))
    desc3_rect = desc3.get_rect(center=(Config.SCREEN_WIDTH//2, 360))
    desc4_rect = desc4.get_rect(center=(Config.SCREEN_WIDTH//2, 420))
    desc5_rect = desc5.get_rect(center=(Config.SCREEN_WIDTH//2, 500))

    screen.blit(desc1, desc1_rect)
    screen.blit(desc2, desc2_rect)
    screen.blit(desc3, desc3_rect)
    screen.blit(desc4, desc4_rect)
    screen.blit(desc5, desc5_rect)

    # 绘制Start Game按钮
    btn_rect = pygame.Rect(Config.SCREEN_WIDTH//2 - 150, 600, 300, 80)
    pygame.draw.rect(screen, Config.BUTTON_BG, btn_rect)
    pygame.draw.rect(screen, Config.BUTTON_BORDER, btn_rect, 3)
    btn_text = font_medium.render("Start Game", True, Config.TEXT_COLOR)
    btn_text_rect = btn_text.get_rect(center=btn_rect.center)
    screen.blit(btn_text, btn_text_rect)

    # 提示文字
    hint_text = font_small.render("Use 'Reset' to restart anytime", True, Config.TEXT_COLOR)
    hint_rect = hint_text.get_rect(center=(Config.SCREEN_WIDTH//2, 720))
    screen.blit(hint_text, hint_rect)

    pygame.display.flip()

    # 显示到Colab
    data = pygame.image.tostring(screen, 'RGB')
    size = screen.get_size()
    img = Image.frombytes('RGB', size, data)
    with out_display:
        IPython.display.clear_output(wait=True)
        IPython.display.display(img)

# 绘制结束界面
def draw_end_screen():
    screen.fill(Config.BG_COLOR)

    # 标题
    end_title = font_large.render("Episode Complete!", True, Config.TEXT_COLOR)
    end_rect = end_title.get_rect(center=(Config.SCREEN_WIDTH//2, 120))
    screen.blit(end_title, end_rect)

    # 本轮统计
    ep_text = font_medium.render(f"Episode: {current_episode}", True, Config.TEXT_COLOR)
    step_text = font_medium.render(f"Total Steps: {current_steps}", True, Config.TEXT_COLOR)
    reward_text = font_medium.render(f"Total Reward: {episode_reward:.1f}", True, Config.TEXT_COLOR)
    mode_text = font_medium.render(f"Mode: {'Training' if training else 'Play'}", True, Config.TEXT_COLOR)

    ep_rect = ep_text.get_rect(center=(Config.SCREEN_WIDTH//2, 250))
    step_rect = step_text.get_rect(center=(Config.SCREEN_WIDTH//2, 350))
    reward_rect = reward_text.get_rect(center=(Config.SCREEN_WIDTH//2, 450))
    mode_rect = mode_text.get_rect(center=(Config.SCREEN_WIDTH//2, 550))

    screen.blit(ep_text, ep_rect)
    screen.blit(step_text, step_rect)
    screen.blit(reward_text, reward_rect)
    screen.blit(mode_text, mode_rect)

    # 提示文字
    continue_text = font_small.render("Click 'Step' for next episode", True, Config.TEXT_COLOR)
    continue_rect = continue_text.get_rect(center=(Config.SCREEN_WIDTH//2, 680))
    screen.blit(continue_text, continue_rect)

    pygame.display.flip()

    # 显示到Colab
    data = pygame.image.tostring(screen, 'RGB')
    size = screen.get_size()
    img = Image.frombytes('RGB', size, data)
    with out_display:
        IPython.display.clear_output(wait=True)
        IPython.display.display(img)

# 绘制游戏主界面
def draw_game_screen():
    screen.fill(Config.BG_COLOR)

    # 迷宫居中计算
    maze_total_width = Config.MAZE_SIZE * Config.CELL_SIZE
    maze_total_height = Config.MAZE_SIZE * Config.CELL_SIZE
    maze_offset_x = (Config.SCREEN_WIDTH - maze_total_width - 300) // 2
    maze_offset_y = (Config.SCREEN_HEIGHT - maze_total_height) // 2

    # 绘制迷宫格子
    for x in range(Config.MAZE_SIZE):
        for y in range(Config.MAZE_SIZE):
            rect = pygame.Rect(
                x*Config.CELL_SIZE + maze_offset_x,
                y*Config.CELL_SIZE + maze_offset_y,
                Config.CELL_SIZE,
                Config.CELL_SIZE
            )
            if maze.is_wall(x, y):
                pygame.draw.rect(screen, (0, 0, 0), rect)
                pygame.draw.rect(screen, (50, 50, 50), rect, 2)
            else:
                pygame.draw.rect(screen, (255, 255, 255), rect)
                pygame.draw.rect(screen, (50, 50, 50), rect, 2)

    # 绘制起点、终点、智能体
    start_rect = pygame.Rect(maze_offset_x, maze_offset_y, Config.CELL_SIZE, Config.CELL_SIZE)
    end_rect = pygame.Rect(
        (Config.MAZE_SIZE-1)*Config.CELL_SIZE + maze_offset_x,
        (Config.MAZE_SIZE-1)*Config.CELL_SIZE + maze_offset_y,
        Config.CELL_SIZE,
        Config.CELL_SIZE
    )
    agent_rect = pygame.Rect(
        agent_x*Config.CELL_SIZE + maze_offset_x + 10,
        agent_y*Config.CELL_SIZE + maze_offset_y + 10,
        Config.CELL_SIZE - 20,
        Config.CELL_SIZE - 20
    )

    pygame.draw.rect(screen, (0, 255, 0), start_rect)  # 起点绿色
    pygame.draw.rect(screen, (255, 0, 0), end_rect)    # 终点红色
    pygame.draw.rect(screen, (0, 0, 255), agent_rect)  # 智能体蓝色

    # 右侧状态栏
    status_x = maze_offset_x + maze_total_width + 50
    status_y = maze_offset_y

    def draw_text(text, x, y, font=font_small):
        text_surface = font.render(text, True, Config.TEXT_COLOR)
        screen.blit(text_surface, (x, y))

    draw_text(f"Episode: {current_episode}/{Config.TOTAL_EPISODES}", status_x, status_y, font_medium)
    draw_text(f"Steps: {current_steps}", status_x, status_y+80, font_medium)
    draw_text(f"Reward: {cumulative_reward:.1f}", status_x, status_y+160, font_medium)
    draw_text(f"Epsilon: {agent.epsilon:.2f}", status_x, status_y+240, font_medium)
    draw_text(f"Mode: {'Training' if training else 'Play'}", status_x, status_y+320, font_medium)

    pygame.display.flip()

    # 显示到Colab
    data = pygame.image.tostring(screen, 'RGB')
    size = screen.get_size()
    img = Image.frombytes('RGB', size, data)
    with out_display:
        IPython.display.clear_output(wait=True)
        IPython.display.display(img)

# 启动游戏（点击Start Game触发）
def start_game_by_ui_button():
    global game_state, current_episode
    if game_state == GAME_STATE_START:
        game_state = GAME_STATE_PLAYING
        current_episode = 1
        draw_game_screen()
        print("🎮 Game started!")

# Step按钮核心逻辑
def step_game(_):
    global agent_x, agent_y, current_episode, current_steps, cumulative_reward, training, game_state, episode_reward

    # 开始界面→游戏中
    if game_state == GAME_STATE_START:
        start_game_by_ui_button()
        return

    # 结束界面→重置开始下一轮
    if game_state == GAME_STATE_END:
        agent_x, agent_y = 0, 0
        current_steps = 0
        cumulative_reward = 0
        episode_reward = 0
        game_state = GAME_STATE_PLAYING
        draw_game_screen()
        return

    # 步数超限→结束本轮
    if current_steps >= Config.MAX_STEPS:
        episode_reward = cumulative_reward
        game_state = GAME_STATE_END
        current_episode += 1
        agent.decay_epsilon()
        draw_end_screen()
        return

    current_steps += 1

    # 训练模式
    if training and current_episode <= Config.TOTAL_EPISODES:
        state = agent.get_state(agent_x, agent_y)
        action = agent.choose_action(state)

        old_x, old_y = agent_x, agent_y
        # 动作执行：0=上，1=下，2=左，3=右
        if action == 0: agent_y -= 1
        elif action == 1: agent_y += 1
        elif action == 2: agent_x -= 1
        elif action == 3: agent_x += 1

        # 撞墙回退
        if maze.is_wall(agent_x, agent_y):
            agent_x, agent_y = old_x, old_y

        # 奖励计算
        if maze.is_wall(agent_x, agent_y):
            reward = -10  # 撞墙惩罚
        elif agent_x == Config.MAZE_SIZE-1 and agent_y == Config.MAZE_SIZE-1:
            reward = 50   # 终点奖励
        else:
            reward = -0.1 # 步数惩罚

        cumulative_reward += reward
        episode_reward = cumulative_reward

        # 更新Q表
        next_state = agent.get_state(agent_x, agent_y)
        agent.update_q_table(state, action, reward, next_state)

        # 到达终点→结束本轮
        if agent_x == Config.MAZE_SIZE-1 and agent_y == Config.MAZE_SIZE-1:
            game_state = GAME_STATE_END
            current_episode += 1
            agent.decay_epsilon()
            draw_end_screen()
            return

    # 游玩模式（仅走最优路径）
    elif not training:
        state = agent.get_state(agent_x, agent_y)
        action = np.argmax(agent.q_table[state])

        old_x, old_y = agent_x, agent_y
        if action == 0: agent_y -= 1
        elif action == 1: agent_y += 1
        elif action == 2: agent_x -= 1
        elif action == 3: agent_x += 1

        if maze.is_wall(agent_x, agent_y):
            agent_x, agent_y = old_x, old_y

        # 奖励计算
        if maze.is_wall(agent_x, agent_y):
            reward = -10
        elif agent_x == Config.MAZE_SIZE-1 and agent_y == Config.MAZE_SIZE-1:
            reward = 50
        else:
            reward = -0.1

        cumulative_reward += reward
        episode_reward = cumulative_reward

        # 到达终点→结束
        if agent_x == Config.MAZE_SIZE-1 and agent_y == Config.MAZE_SIZE-1:
            game_state = GAME_STATE_END
            draw_end_screen()
            return

    draw_game_screen()

# 切换训练/游玩模式
def toggle_mode(_):
    global training
    training = not training
    if game_state == GAME_STATE_PLAYING:
        draw_game_screen()

# 重置游戏
def reset_game(_):
    global agent_x, agent_y, current_episode, current_steps, cumulative_reward, training, game_state, maze, agent, episode_reward
    maze = Maze(Config.MAZE_SIZE, Config.WALL_DENSITY)
    agent = QLearningAgent(Config.MAZE_SIZE, Config.LEARNING_RATE, Config.DISCOUNT_FACTOR)
    agent_x, agent_y = 0, 0
    current_episode = 0
    current_steps = 0
    cumulative_reward = 0
    episode_reward = 0
    training = True
    game_state = GAME_STATE_START
    draw_start_screen()

# 创建交互按钮
btn_step = Button(description="Step (1 step)", style={'button_color': '#4CAF50'}, layout=Layout(width='200px', height='60px', font_size='20px'))
btn_mode = Button(description="Train/Play", style={'button_color': '#2196F3'}, layout=Layout(width='200px', height='60px', font_size='20px'))
btn_reset = Button(description="Reset", style={'button_color': '#f44336'}, layout=Layout(width='200px', height='60px', font_size='20px'))

# 绑定按钮事件
btn_step.on_click(step_game)
btn_mode.on_click(toggle_mode)
btn_reset.on_click(reset_game)

# 按钮布局（靠左）
display_ui = HBox([btn_step, btn_mode, btn_reset], layout=Layout(justify_content='flex-start', height='80px', padding='0 20px'))
IPython.display.display(display_ui)
IPython.display.display(out_display)

# 初始绘制开始界面
draw_start_screen()