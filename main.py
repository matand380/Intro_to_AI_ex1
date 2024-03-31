
import sys
import algorithms as alg

# Update the recursion limit for DFS, IDS, and IDA*
sys.setrecursionlimit(100000)


def main():
    # Read the input data from the file
    with open("input.txt", "r") as input_file:  # Adjust the path as necessary
        algorithm = input_file.readline().strip().upper()
        board_size = int(input_file.readline().strip())
        initial_state = list(map(int, input_file.readline().strip().split(',')))

    goal_state = list(range(1, board_size * board_size)) + [0]

    if algorithm == "IDS":
        solution = alg.ids_solver(initial_state, goal_state, lambda state: alg.actions(state, board_size))
    elif algorithm == "BFS":
        solution = alg.bfs_solver(initial_state, goal_state, lambda state: alg.actions(state, board_size))
    elif algorithm == "A*":
        solution = alg.a_star_solver(initial_state, goal_state,
                                     lambda state: alg.actions(state, board_size),
                                     lambda state, goal=goal_state: alg.manhattan_distance(state, goal, board_size))

    elif algorithm == "DFS":
        solution = alg.dfs_solver(100, initial_state, goal_state, lambda state: alg.actions(state, board_size),
                                         [initial_state])
    elif algorithm == "IDA*":
        solution = alg.ida_star_solver(initial_state, goal_state, lambda state: alg.actions(state, board_size),
                                       lambda state: alg.manhattan_distance(state, goal_state, board_size))
    else:
        raise ValueError(f"Algorithm {algorithm} is not supported.")

    # Ensure a solution was found before attempting to trace actions
    if solution:
        actions_path = alg.find_actions_path(solution)
        with open("output.txt", "w") as output_file:  # Adjust the path as necessary
            output_file.write(" ".join(actions_path))
        print("Done")
    else:
        print("No solution found.")


if __name__ == "__main__":
    main()
