Maze Q-Learning
This project uses the Q-Learning reinforcement learning algorithm to design and implement a maze-solving game. In this game, an AI agent learns to find the optimal path from the start point to the end point in a grid maze through continuous exploration and reward feedback.

Features
Two game modes:
  Training Mode: The agent uses the ε-greedy strategy to explore the maze and update the Q-value table.
  Play Mode: The agent uses the trained Q-table to select optimal actions and solve the maze.
Visualized interface: Clear display of the maze, agent, start/end points, and real-time training status.
Interactive controls: Step-by-step execution, mode switching, and game reset via buttons.
Cloud environment support: Adapted to run in Colab with virtual display and image-based UI rendering.

How to Run
1.Open the project in a Python environment (e.g., Google Colab).
2.Run the code to install dependencies and initialize the game.
3.Click the Step (1 step) button to start the game and let the agent explore.
4.Use Train/Play to switch between training and play modes.
5.Click Reset to restart the game with a new maze.

Game Rules
The agent (blue square) starts at the top-left corner (0, 0) of a 5×5 maze.
The goal is to reach the red end point at the bottom-right corner.
The agent can move up, down, left, or right, but cannot pass through walls.
Reward mechanism:
  +50 for reaching the end point.
  -10 for colliding with walls or maze boundaries.
  -0.1 for each step taken to encourage finding the shortest path.

Algorithm Implementation
The core of the project is the Q-Learning algorithm:
1.Q-table: A 2D array initialized to 0, where rows represent states (maze positions) and columns represent actions (movement directions).
2.ε-greedy strategy: Balances exploration (random actions) and exploitation (optimal actions based on Q-values).
3.Q-value update: Uses the formula:Q(s,a)=Q(s,a)+α×[r+γ×maxQ(s′,a′)−Q(s,a)]where α is the learning rate, γ is the discount factor, r is the immediate reward, and maxQ(s′,a′) is the maximum Q-value of the next state.
