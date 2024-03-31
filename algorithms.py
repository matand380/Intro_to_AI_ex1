import heapq
from collections import deque
from itertools import count

"""
Puzzle Solving Algorithms Script

Overview: This script addresses the challenge of solving sliding tile puzzles (e.g., the 8-puzzle, 15-puzzle) by 
implementing a suite of search algorithms. Each algorithm offers a unique strategy for exploring the space of 
possible configurations to transition from a given initial state to a designated goal state. The collection of 
algorithms provides various approaches, accommodating different puzzle characteristics and solution requirements.

Algorithms Implemented:

1. Iterative Deepening Search (IDS) - Merges the space efficiency of depth-first exploration 
with the systematic level-by-level search characteristic of breadth-first search. By progressively deepening the 
search depth, IDS effectively balances the strengths of both DFS and BFS.

2. Breadth-First Search (BFS) - Explores the state space systematically, one level at a time, guaranteeing that the 
shortest path to the goal (if one exists) is found. BFS is optimal for puzzles with uniform step costs and is 
complete, ensuring solution discovery if one exists.

3. A* Search - Employs heuristics to estimate the cost to reach the goal from the current state, prioritizing paths 
that appear closer to the goal. A* is both complete and optimal, provided the heuristic is admissible, meaning it 
never overestimates the actual cost to reach the goal.

4. Depth-First Search (DFS) - Dives deep into each branch before backtracking, offering a memory-efficient 
alternative to BFS. However, DFS does not guarantee the shortest path and may not find a solution in puzzles with 
infinite or very deep states without imposing a depth limit.

5. Iterative Deepening A* (IDA*) - Combines the depth-limited search strategy of IDS with the heuristic-driven 
prioritization of A*. IDA* iteratively increases the cost threshold, using the heuristic to explore the most 
promising paths first. Like A*, it is complete and optimal for admissible heuristics, with the added benefit of 
reduced memory usage similar to IDS.

Usage: Define the puzzle's initial state and goal state, then select an appropriate solving algorithm based on the 
puzzle's characteristics and solution requirements. The script will attempt to find a solution path from the initial 
to the goal state, detailing the required moves to achieve the goal configuration."""


def actions(state, n):
    """
    Generate all possible next states from the current state, ordered by Up, Down, Left, Right.

    Args:
        state (list): The current state of the puzzle, represented as a flat list.
        n (int): The size of the puzzle grid.

    Returns:
        list: Ordered list of all possible next states.
    """
    # Find the index of the empty space (denoted by 0) in the puzzle.
    i = state.index(0)
    # Calculate the row and column of the empty space based on its index.
    row, col = divmod(i, n)  # div mod returns the quotient and the remainder
    # Initialize a list to hold the positions of the neighbors in the specified order.
    ordered_neighbors = []
    # Define the order of movement: Up, Down, Left, Right, as tuples indicating row and column changes.
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Iterate over each direction to calculate the new positions of the empty space after the move.
    for dr, dc in directions:
        # Calculate the new row and column by applying the direction to the current row and column.
        new_row, new_col = row + dr, col + dc
        # Check if the new position is within the bounds of the puzzle grid.
        if 0 <= new_row < n and 0 <= new_col < n:
            # If valid, append the new position to the list of neighbors.
            ordered_neighbors.append((new_row, new_col))
    # Function to swap the empty space with a neighbor to create a new state.

    def swap(a, b):
        # Create a copy of the current state to modify.
        new_state = state.copy()
        # Perform the swap: move the empty space to the new position.
        new_state[a], new_state[b] = new_state[b], new_state[a]
        # Return the new state after the move.
        return new_state
    # Generate and return all possible states from the current state by applying the swap for each neighbor.
    return [swap(i, new_row * n + new_col) for new_row, new_col in ordered_neighbors]


def manhattan_distance(state, goal_state, n):
    """
    Calculate the Manhattan distance heuristic for a given state.

    The Manhattan distance is the sum of the absolute differences of the coordinates.
    It represents the minimum number of moves required to move each tile from its
    position in the current state to its position in the goal state, moving only
    horizontally or vertically.

    Args:
        state (list): The current state of the puzzle, represented as a flat list.
        goal_state (list): The goal state of the puzzle, also as a flat list.
        n (int): The size of the puzzle grid, indicating it's an n x n puzzle.

    Returns:
        int: The total Manhattan distance from the current state to the goal state.
    """
    # Initialize the total distance to 0.
    distance = 0
    # Iterate over each tile in the puzzle.
    for i in range(n * n):
        # Get the value of the current tile, excluding the empty space (denoted by 0).
        current_tile = state[i]
        # If the current tile is not the empty space:
        if current_tile != 0:
            # Find the goal position (row and column) of the current tile in the goal state.
            # The `div mod` function is used here to get the row (quotient) and column (remainder)
            # based on the tile's index in the goal state.
            goal_i, goal_j = divmod(goal_state.index(current_tile), n)
            # Calculate the Manhattan distance for the current tile:
            # The difference in rows (`i // n - goal_i`) plus the difference in columns (`i % n - goal_j`).
            # `i // n` gives the current row of the tile, `i % n` gives the current column.
            # The `abs` function is used to ensure the distance is positive.
            distance += abs(i // n - goal_i) + abs(i % n - goal_j)
    # Return the total Manhattan distance for the puzzle state.
    return distance


def find_actions_path(solution):
    """
    Finds the sequence of actions that leads from the initial state to the goal state
    based on the positions of the empty tile (0) in each state.
    """

    actions_path = []  # Initialize an empty list to store the sequence of actions.

    # Loop through the solution path, starting from the second state (index 1)
    # because we are comparing each state with its predecessor.
    for i in range(1, len(solution)):
        # prev_state is the state before the current state in the solution path.
        # current_state is the current state being analyzed.
        prev_state, current_state = solution[i - 1], solution[i]

        # Calculate the difference in the index of the empty tile (0) between
        # the current state and the previous state to determine the direction of the move.
        diff = current_state.index(0) - prev_state.index(0)

        # Determine the action taken based on the index difference:
        if diff == 1:
            # If diff is 1, the empty tile moved left in the grid representation.
            actions_path.append("L")
        elif diff == -1:
            # If diff is -1, the empty tile moved right in the grid representation.
            actions_path.append("R")
        elif diff > 0:
            # If diff is greater than 0 (and not 1, which is handled above),
            # the empty tile moved up. The difference is positive and corresponds
            # to a vertical move in the grid, indicating an upward movement.
            actions_path.append("U")
        else:
            # If none of the above conditions are true, the only remaining
            # option is that the empty tile moved down in the grid.
            actions_path.append("D")

    # Return the complete list of actions that represent the moves taken
    # from the initial state to reach the goal state.
    return actions_path


def ids_solver(state, goal_state, actions):
    """
    Solve the puzzle using Iterative Deepening Search (IDS).

    Args:
        state (list): The initial state of the puzzle.
        goal_state (list): The goal state of the puzzle.
        actions (function): Function that generates all possible next states from the current state.

    Returns:
        list: The solution path from the initial state to the goal state, or None if no solution is found.
    """
    # Iteratively deepen the search starting from depth 0.
    for depth in count(start=1):
        result = dfs_solver(depth, state, goal_state, actions, [state])
        # If a solution is found at the current depth, return it.
        if result:
            return result


def bfs_solver(state, goal_state, actions):
    """
    Solve the puzzle using the Breadth-First Search (BFS) algorithm more efficiently.

    Args:
        state (list): The initial state of the puzzle, represented as a flat list.
        goal_state (list): The goal state of the puzzle, also as a flat list.
        actions (function): A function that generates all possible next states from the current state.

    Returns:
        list: The solution path from the initial state to the goal state, or None if no solution is found.
    """
    # Initialize a deque with a tuple containing the initial state and the path taken to reach it.
    queue = deque([(state, [state])])
    # Create a set of visited states to prevent revisiting the same state.
    visited = {tuple(state)}
    # Loop as long as there are states in the queue to explore.
    while queue:
        # Remove and return the leftmost state from the queue to explore its successors.
        current_state, path = queue.popleft()
        # Check if the current state is the goal state.
        if current_state == goal_state:
            return path  # Return the path taken to reach the goal state if it is.
        # Generate all possible next states from the current state.
        for new_state in actions(current_state):
            new_state_tuple = tuple(new_state)  # Convert to tuple for hash ability and set operations.
            # If the new state has not been visited, add it to the queue and mark it as visited.
            if new_state_tuple not in visited:
                visited.add(new_state_tuple)  # Mark the new state as visited.
                # Append the new state to the queue with the updated path.
                queue.append((new_state, path + [new_state]))
    # If the function exhausts all possibilities without finding the goal state, return None.
    return None


def a_star_solver(state, goal_state, actions, heuristic):
    """
    Solve the puzzle using A* Search algorithm.

    Args:
        state (list): The initial state of the puzzle.
        goal_state (list): The goal state of the puzzle.
        actions (function): The function to generate possible actions for a given state.
        heuristic (function): The heuristic function to estimate the cost to reach the goal state.

    Returns:
        list: The solution path from the initial state to the goal state, or None if no solution is found.
    """
    # Initialize the priority queue with a tuple containing the heuristic value, current state, and path.
    open_list = [(heuristic(state, goal_state), state, [state])]
    # Initialize the set of visited states to avoid revisiting the same state.
    visited = {tuple(state)}

    # Continue searching while there are states in the priority queue.
    while open_list:
        # Pop the state with the lowest total cost from the priority queue.
        _, current_state, path = heapq.heappop(open_list)
        # Check if the current state is the goal state.
        if current_state == goal_state:
            return path  # Return the path if the goal state is reached.
        # Generate all possible next states from the current state.
        for new_state in actions(current_state):
            # Check if the new state has not been visited before.
            if tuple(new_state) not in visited:
                # Mark the new state as visited.
                visited.add(tuple(new_state))
                # Create a new path by appending the new state to the current path.
                new_path = path + [new_state]
                # Calculate the total cost for the new path using the heuristic.
                total_cost = len(new_path) + heuristic(new_state, goal_state)
                # Add the new state, its total cost, and the new path to the priority queue.
                heapq.heappush(open_list, (total_cost, new_state, new_path))

    # If no solution is found, return None.
    return None


# Depth,First Search (DFS) with Depth Limit
def dfs_solver(depth, state, goal_state, actions, path):
    """
    Depth-First Search function with a depth limit.

    Args:
        depth (int): The current depth limit.
        state (list): The current state of the puzzle.
        goal_state (list): The goal state of the puzzle.
        actions (function): Function to generate possible actions.
        path (list): The current path taken to reach the state.

    Returns:
        list: The solution path if found, None otherwise.
    """
    # Base case: if current state is goal state, return the path.
    if state == goal_state:
        return path
    # Base case: if depth limit reached, terminate this path.
    if depth == 0:
        return None
    # Iterate over all possible next states from the current state.
    for action in actions(state):
        new_state = action
        # Avoid revisiting states in the current path.
        if new_state not in path:
            # Recursively call dfs with reduced depth and new state.
            solution = dfs_solver(depth - 1, new_state, goal_state, actions, path + [new_state])
            # If a solution is found in the recursion, return it.
            if solution:
                return solution
    # If no solution is found at this depth, return None.
    return None


# IDA* Search
def ida_star_solver(state, goal_state, actions, heuristic):
    """
    Solve the puzzle using Iterative Deepening A* (IDA*) algorithm.

    Args:
        state (list): The initial state of the puzzle.
        goal_state (list): The goal state of the puzzle.
        actions (function): The function to generate possible actions for a given state.
        heuristic (function): The heuristic function to estimate the cost to reach the goal state.

    Returns:
        list: The solution path from the initial state to the goal state, or None if no solution is found.
    """

    def search(path, g, threshold):
        """
        Depth-limited search function for IDA* algorithm.

        Args:
            path (list): The current path of states from the initial state to the current state.
            g (int): The cost to reach the current state from the initial state.
            threshold (int): The threshold for searching states based on the total estimated cost.

        Returns:
            list or int: If a solution path is found, returns the path; otherwise, returns the new threshold.
        """
        current_state = path[-1]  # Get the current state from the end of the path.
        f = g + heuristic(current_state)  # Calculate the total estimated cost.
        # If the total estimated cost exceeds the threshold, return the cost.
        if f > threshold:
            return f
        # If the current state is the goal state, return the solution path.
        if current_state == goal_state:
            return path
        min_threshold = float('inf')  # Initialize the minimum threshold.
        # Iterate over all possible successor states from the current state.
        for successor in actions(current_state):
            if successor not in path:  # Ensure successor state is not already visited.
                path.append(successor)  # Add the successor state to the path.
                t = search(path, g + 1, threshold)  # Recursively search the successor state.
                # If a solution path is found, return it.
                if isinstance(t, list):
                    return t
                # Update the minimum threshold based on the returned value.
                if t < min_threshold:
                    min_threshold = t
                path.pop()  # Remove the last state to backtrack.
        return min_threshold  # Return the minimum threshold for the next iteration.

    threshold = heuristic(state)  # Set the initial threshold based on the heuristic estimate.
    path = [state]  # Initialize the path with the initial state.
    # Iteratively deepen the search until a solution path is found.
    while True:
        t = search(path, 0, threshold)  # Perform depth-limited search with the current threshold.
        # If a solution path is found, return it.
        if isinstance(t, list):
            return t
        # If the threshold reaches infinity, indicating no solution path is possible, return None.
        if t == float('inf'):
            return None
        threshold = t  # Update the threshold for the next iteration.
