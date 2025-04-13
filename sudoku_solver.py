import streamlit as st
import time
import copy
import random

# -------------------------------
# Standalone helper: Check validity for a given board cell.
# -------------------------------
def is_valid(board, row, col, num):
    # Row check
    for c in range(9):
        if board[row][c] == num:
            return False
    # Column check
    for r in range(9):
        if board[r][col] == num:
            return False
    # 3x3 Box check
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if board[r][c] == num:
                return False
    return True

# -------------------------------
# SudokuSolver Class
# -------------------------------
class SudokuSolver:
    def __init__(self, board):
        self.board = board
        self.iterations = 0  # Counter for recursive calls

    def find_empty(self, board):
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    return (r, c)
        return None

    def is_valid(self, board, row, col, num):
        # Row
        for c in range(9):
            if board[row][c] == num:
                return False
        # Column
        for r in range(9):
            if board[r][col] == num:
                return False
        # 3x3 Box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if board[r][c] == num:
                    return False
        return True

    def solve_backtracking(self, board):
        self.iterations += 1
        empty = self.find_empty(board)
        if not empty:
            return True  # Solved
        row, col = empty
        for num in range(1, 10):
            if self.is_valid(board, row, col, num):
                board[row][col] = num
                if self.solve_backtracking(board):
                    return True
                board[row][col] = 0  # Backtrack
        return False

    def solve_backtracking_random(self, board):
        """Variant of backtracking for generating a complete board.
        Randomizes the order of numbers to achieve variety."""
        self.iterations += 1
        empty = self.find_empty(board)
        if not empty:
            return True
        row, col = empty
        numbers = list(range(1, 10))
        random.shuffle(numbers)
        for num in numbers:
            if self.is_valid(board, row, col, num):
                board[row][col] = num
                if self.solve_backtracking_random(board):
                    return True
                board[row][col] = 0
        return False

    def get_neighbors(self, row, col):
        """Return positions that share a row, column, or 3x3 box with the (row, col) cell."""
        neighbors = set()
        for i in range(9):
            if i != col:
                neighbors.add((row, i))
            if i != row:
                neighbors.add((i, col))
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(start_row, start_row + 3):
            for c in range(start_col, start_col + 3):
                if (r, c) != (row, col):
                    neighbors.add((r, c))
        return neighbors

    def init_domains(self, board):
        """Initialize domains for every cell based on the current board."""
        domains = {}
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    domains[(r, c)] = set(range(1, 10))
                else:
                    domains[(r, c)] = {board[r][c]}
        # Prune domains based on pre-assigned values
        for r in range(9):
            for c in range(9):
                if board[r][c] != 0:
                    num = board[r][c]
                    for neighbor in self.get_neighbors(r, c):
                        nr, nc = neighbor
                        if board[nr][nc] == 0 and num in domains[(nr, nc)]:
                            domains[(nr, nc)].remove(num)
        return domains

    def solve_forward_checking(self, board, domains):
        self.iterations += 1
        # Find an empty cell; also choose one with Minimum Remaining Values (MRV)
        empty = None
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    empty = (r, c)
                    break
            if empty:
                break
        if not empty:
            return True  # Solved

        # Use MRV to select the unassigned cell with the smallest domain.
        row, col = empty
        min_size = 10
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0 and len(domains[(r, c)]) < min_size:
                    min_size = len(domains[(r, c)])
                    row, col = r, c

        # Try all possible numbers from the selected cell’s domain.
        for num in sorted(domains[(row, col)]):
            if self.is_valid(board, row, col, num):
                board[row][col] = num
                new_domains = copy.deepcopy(domains)
                new_domains[(row, col)] = {num}
                failed = False
                for neighbor in self.get_neighbors(row, col):
                    nr, nc = neighbor
                    if board[nr][nc] == 0 and num in new_domains[(nr, nc)]:
                        new_domains[(nr, nc)].remove(num)
                        if not new_domains[(nr, nc)]:
                            failed = True
                            break
                if failed:
                    board[row][col] = 0
                    continue
                if self.solve_forward_checking(board, new_domains):
                    return True
                board[row][col] = 0  # Backtrack
        return False

# -------------------------------
# Uniqueness checker: Count number of solutions.
# -------------------------------
def count_solutions(board):
    count = 0

    def solve_count(bd):
        nonlocal count
        empty = None
        for r in range(9):
            for c in range(9):
                if bd[r][c] == 0:
                    empty = (r, c)
                    break
            if empty:
                break
        if not empty:
            count += 1
            return
        row, col = empty
        for num in range(1, 10):
            if is_valid(bd, row, col, num):
                bd[row][col] = num
                solve_count(bd)
                if count > 1:  # Early exit if more than one solution is found.
                    return
                bd[row][col] = 0

    board_copy = copy.deepcopy(board)
    solve_count(board_copy)
    return count

# -------------------------------
# Generate a complete sudoku board
# -------------------------------
def generate_complete_board():
    board = [[0 for _ in range(9)] for _ in range(9)]
    solver = SudokuSolver(board)
    solver.solve_backtracking_random(board)
    return board

# -------------------------------
# Generate a puzzle with unique solution based on difficulty.
# -------------------------------
def generate_puzzle(difficulty):
    complete_board = generate_complete_board()
    # Set the number of clues (filled cells) based on difficulty.
    if difficulty == "Easy":
        clues = 40
    elif difficulty == "Medium":
        clues = 32
    elif difficulty == "Hard":
        clues = 25
    cells_to_remove = 81 - clues

    # Remove cells in a random order, ensuring puzzle uniqueness.
    positions = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(positions)
    puzzle = copy.deepcopy(complete_board)
    removed = 0
    for (r, c) in positions:
        if removed >= cells_to_remove:
            break
        temp = puzzle[r][c]
        puzzle[r][c] = 0
        # Check if the puzzle still has a unique solution.
        if count_solutions(puzzle) != 1:
            puzzle[r][c] = temp  # Revert removal if not unique.
        else:
            removed += 1
    return puzzle

# -------------------------------
# Helper function: Display sudoku board in Streamlit.
# -------------------------------
def display_board(board, header):
    st.subheader(header)
    # Display as rows in a table-like fashion.
    for row in board:
        st.write(row)

# -------------------------------
# Main Streamlit App
# -------------------------------
st.title("Sudoku Solver with Random Puzzle Generator")

# Sidebar: User selections
difficulty = st.sidebar.selectbox("Select Puzzle Difficulty", ["Easy", "Medium", "Hard"])
solver_method = st.sidebar.selectbox("Select Solving Method", ["Backtracking", "Forward Checking"])
test_runs = st.sidebar.number_input("Number of Test Runs", min_value=1, value=5, step=1)

# Use session_state to hold the puzzle and its solution
if "puzzle" not in st.session_state:
    st.session_state["puzzle"] = None
if "solution" not in st.session_state:
    st.session_state["solution"] = None

# Button to generate a new puzzle
if st.sidebar.button("Generate Puzzle"):
    with st.spinner("Generating puzzle..."):
        puzzle = generate_puzzle(difficulty)
        st.session_state["puzzle"] = puzzle
        st.session_state["solution"] = None
    st.success("Puzzle generated!")

# Display the generated puzzle (if any)
if st.session_state["puzzle"] is not None:
    display_board(st.session_state["puzzle"], "Generated Puzzle")

# Button to solve the puzzle using the selected method and report performance.
if st.sidebar.button("Solve Puzzle"):
    if st.session_state["puzzle"] is None:
        st.error("Please generate a puzzle first!")
    else:
        puzzle = copy.deepcopy(st.session_state["puzzle"])
        total_time = 0
        total_iters = 0
        solved_board = None
        for _ in range(test_runs):
            board_copy = copy.deepcopy(puzzle)
            solver = SudokuSolver(board_copy)
            start_time = time.time()
            if solver_method == "Backtracking":
                solver.solve_backtracking(board_copy)
            else:
                domains = solver.init_domains(board_copy)
                solver.solve_forward_checking(board_copy, domains)
            elapsed = time.time() - start_time
            total_time += elapsed
            total_iters += solver.iterations
            solved_board = board_copy
        avg_time = total_time / test_runs
        avg_iters = total_iters / test_runs
        st.write(f"**Solving Method:** {solver_method}")
        st.write(f"**Average Time:** {avg_time:.4f} seconds over {test_runs} runs")
        st.write(f"**Average Iterations:** {avg_iters:.0f}")
        st.session_state["solution"] = solved_board
        display_board(solved_board, "Solved Puzzle")

# Button to compare performance of both methods on the same board.
if st.sidebar.button("Compare Performance (Both Methods)"):
    if st.session_state["puzzle"] is None:
        st.error("Please generate a puzzle first!")
    else:
        puzzle = copy.deepcopy(st.session_state["puzzle"])
        methods = ["Backtracking", "Forward Checking"]
        results = []
        for method in methods:
            total_time = 0
            total_iters = 0
            for _ in range(test_runs):
                board_copy = copy.deepcopy(puzzle)
                solver = SudokuSolver(board_copy)
                start_time = time.time()
                if method == "Backtracking":
                    solver.solve_backtracking(board_copy)
                else:
                    domains = solver.init_domains(board_copy)
                    solver.solve_forward_checking(board_copy, domains)
                elapsed = time.time() - start_time
                total_time += elapsed
                total_iters += solver.iterations
            avg_time = total_time / test_runs
            avg_iters = total_iters / test_runs
            results.append((method, avg_time, avg_iters))
        st.subheader("Performance Comparison")
        for res in results:
            st.write(f"**{res[0]}** – Avg Time: {res[1]:.4f}s, Avg Iterations: {res[2]:.0f}")
