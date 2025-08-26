import random

class GridWorld:
    def __init__(self):
        self.grid_size = (4, 5)
        self.initial_state = (0, 0)
        self.terminal_state = (3, 3)

        # impassable walls.
        self.walls = [(1, 1), (2, 2)]

        # Traps cause heavy negative reward (can be randomized).
        self.traps = {(3, 0): -10, (0, 4): -5}

        # Teleporters: instant jump.
        self.teleporters = {(1, 2): (3, 0)}

        # Movements
        self.actions = {
            'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1),
            'up_left': (-1, -1), 'up_right': (-1, 1),
            'down_left': (1, -1), 'down_right': (1, 1)
        }

    def is_terminal(self, state):
        return state == self.terminal_state

    def get_next_state(self, state, action):
        row, col = state
        d_row, d_col = self.actions[action]
        new_row = row + d_row
        new_col = col + d_col
        new_state = (new_row, new_col)

        if not (0 <= new_row < self.grid_size[0]) or not (0 <= new_col < self.grid_size[1]):
            return state  # Outside boundary
        if new_state in self.walls:
            return state  # Hit a wall.
        if new_state in self.teleporters:
            return self.teleporters[new_state]  # Teleport.
        return new_state

    def get_reward(self, state):
        if state == self.terminal_state:
            return +10
        elif state in self.traps:
            return self.traps[state]
        else:
            return -0.5  # Small step penalty.

    def get_all_states(self):
        return [
            (r, c)
            for r in range(self.grid_size[0])
            for c in range(self.grid_size[1])
            if (r, c) not in self.walls
        ]

    def randomize_traps(self):
        # Randomly change trap positions each episode.
        possible_traps = [
            (r, c)
            for r in range(self.grid_size[0])
            for c in range(self.grid_size[1])
            if (r, c) not in self.walls and (r, c) != self.terminal_state
        ]
        self.traps = {random.choice(possible_traps): -10}
        print(f"New trap set at {list(self.traps.keys())[0]}")


    def print_grid(self, agent_pos=None):
        print("\n🗺️  Grid Snapshot:")
        for r in range(self.grid_size[0]):
            row_cells = []
            for c in range(self.grid_size[1]):
                cell = (r, c)
                if cell == agent_pos:
                    row_cells.append(" A ")  # Agent
                elif cell == self.terminal_state:
                    row_cells.append(" T ")  # Terminal
                elif cell in self.walls:
                    row_cells.append(" # ")  # Wall
                elif cell in self.traps:
                    row_cells.append(" X ")  # Trap
                elif cell in self.teleporters:
                    row_cells.append(" O ")  # Teleporter
                else:
                    row_cells.append(" . ")  # Empty space
            print("".join(row_cells))