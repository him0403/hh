import streamlit as st
import time
import copy
import random
import pandas as pd

# ----------------------------------------------------
# Utility function: Check if placing 'num' is valid.
# ----------------------------------------------------
def is_valid(board, row, col, num):
    # Check the row
    for c in range(9):
        if board[row][c] == num:
            return False
    # Check the column
    for r in range(9):
        if board[r][col] == num:
            return False
    # Check the 3x3 sub-grid
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if board[r][c] == num:
                return False
    return True

# ----------------------------------------------------
# SudokuSolver class with different solving methods.
# ----------------------------------------------------
class SudokuSolver:
    def __init__(self, board):
        self.board = board
        self.iterations = 0  # Counts recursive calls

    def find_empty(self, board):
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    return (r, c)
        return None

    def is_valid(self, board, row, col, num):
        return is_valid(board, row, col, num)

    # --------------------------
    # Basic Backtracking Solver
    # --------------------------
    def solve_backtracking(self, board):
        self.iterations += 1
        empty = self.find_empty(board)
        if not empty:
            return True  # Solved!
        row, col = empty
        for num in range(1, 10):
            if self.is_valid(board, row, col, num):
                board[row][col] = num
                if self.solve_backtracking(board):
                    return True
                board[row][col] = 0  # Backtrack
        return False

    # -----------------------------------------------
    # Randomized Backtracking (for complete board gen.)
    # -----------------------------------------------
    def solve_backtracking_random(self, board):
        self.iterations += 1
        empty = self.find_empty(board)
        if not empty:
            return True
        row, col = empty
        nums = list(range(1, 10))
        random.shuffle(nums)
        for num in nums:
            if self.is_valid(board, row, col, num):
                board[row][col] = num
                if self.solve_backtracking_random(board):
                    return True
                board[row][col] = 0
        return False

    # -----------------------------------------
    # Forward Checking Solver (with MRV heuristic)
    # -----------------------------------------
    def solve_forward_checking(self, board, domains):
        self.iterations += 1
        # Choose an empty cell using MRV (minimum remaining values)
        empty = None
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    empty = (r, c)
                    break
            if empty:
                break
        if not empty:
            return True
        row, col = empty
        min_size = 10
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0 and len(domains[(r, c)]) < min_size:
                    min_size = len(domains[(r, c)])
                    row, col = r, c

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
                board[row][col] = 0
        return False

    # --------------------------
    # CSP solver: Combines AC3, MRV, Degree Tie-breaker, and LCV ordering
    # --------------------------
    def solve_csp(self, board, domains):
        self.iterations += 1
        # Enforce arc consistency
        domains = self.ac3(domains)
        if domains is False:
            return False
        # Check if assignment is complete.
        complete = True
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    complete = False
                    break
            if not complete:
                break
        if complete:
            return True

        # Select unassigned variable using MRV with Degree heuristic.
        var = self.select_unassigned_variable(board, domains)
        if var is None:
            return True
        row, col = var
        # Order values by Least Constraining Value (LCV)
        for val in self.order_domain_values(row, col, board, domains):
            if self.is_valid(board, row, col, val):
                board[row][col] = val
                new_domains = copy.deepcopy(domains)
                new_domains[(row, col)] = {val}
                if self.solve_csp(board, new_domains):
                    return True
                board[row][col] = 0
        return False

    # --------------------------------------------------
    # Helper: Get neighbors (same row, column, and box)
    # --------------------------------------------------
    def get_neighbors(self, row, col):
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

    # ---------------------------
    # Initialize domains for each cell.
    # ---------------------------
    def init_domains(self, board):
        domains = {}
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    domains[(r, c)] = set(range(1, 10))
                else:
                    domains[(r, c)] = {board[r][c]}
        # Prune domains based on assigned cells.
        for r in range(9):
            for c in range(9):
                if board[r][c] != 0:
                    num = board[r][c]
                    for neighbor in self.get_neighbors(r, c):
                        nr, nc = neighbor
                        if board[nr][nc] == 0 and num in domains[(nr, nc)]:
                            domains[(nr, nc)].remove(num)
        return domains

    # -----------------------------------------
    # AC3: Enforce arc-consistency.
    # -----------------------------------------
    def ac3(self, domains):
        queue = [(xi, xj) for xi in domains for xj in self.get_neighbors(xi[0], xi[1])]
        while queue:
            (xi, xj) = queue.pop(0)
            if self.revise(domains, xi, xj):
                if len(domains[xi]) == 0:
                    return False
                for xk in self.get_neighbors(xi[0], xi[1]):
                    if xk != xj:
                        queue.append((xk, xi))
        return domains

    def revise(self, domains, xi, xj):
        revised = False
        for a in set(domains[xi]):
            # For the constraint "≠", if every value in domain[xj] equals a then a cannot be used.
            if all(b == a for b in domains[xj]):
                domains[xi].remove(a)
                revised = True
        return revised

    # --------------------------------------------------
    # MRV with Degree Tie-breaker: Select unassigned variable.
    # --------------------------------------------------
    def select_unassigned_variable(self, board, domains):
        candidate = None
        min_size = 10
        max_degree = -1
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    size = len(domains[(r, c)])
                    if size < min_size:
                        min_size = size
                        candidate = (r, c)
                        max_degree = sum(1 for n in self.get_neighbors(r, c) if board[n[0]][n[1]] == 0)
                    elif size == min_size:
                        degree = sum(1 for n in self.get_neighbors(r, c) if board[n[0]][n[1]] == 0)
                        if degree > max_degree:
                            candidate = (r, c)
                            max_degree = degree
        return candidate

    # --------------------------------------------------
    # LCV: Order domain values by least constraining effect.
    # --------------------------------------------------
    def order_domain_values(self, row, col, board, domains):
        values = list(domains[(row, col)])
        def count_constraints(val):
            count = 0
            for (nr, nc) in self.get_neighbors(row, col):
                if board[nr][nc] == 0 and val in domains[(nr, nc)]:
                    count += 1
            return count
        values.sort(key=lambda val: count_constraints(val))
        return values

# ----------------------------------------------------
# Puzzle Generator Functions
# ----------------------------------------------------
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
                if count > 1:
                    return
                bd[row][col] = 0
    board_copy = copy.deepcopy(board)
    solve_count(board_copy)
    return count

def generate_complete_board():
    board = [[0 for _ in range(9)] for _ in range(9)]
    solver = SudokuSolver(board)
    solver.solve_backtracking_random(board)
    return board

def generate_puzzle(difficulty):
    complete_board = generate_complete_board()
    # Set clues based on difficulty.
    if difficulty == "Easy":
        clues = 40
    elif difficulty == "Medium":
        clues = 32
    elif difficulty == "Hard":
        clues = 25
    cells_to_remove = 81 - clues
    positions = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(positions)
    puzzle = copy.deepcopy(complete_board)
    removed = 0
    for (r, c) in positions:
        if removed >= cells_to_remove:
            break
        temp = puzzle[r][c]
        puzzle[r][c] = 0
        if count_solutions(puzzle) != 1:
            puzzle[r][c] = temp
        else:
            removed += 1
    return puzzle

# ----------------------------------------------------
# Display the board in grid format using a pandas DataFrame.
# ----------------------------------------------------
def display_board_grid(board, header):
    st.subheader(header)
    df = pd.DataFrame(board, columns=[str(i + 1) for i in range(9)])
    st.table(df)

# ----------------------------------------------------
# Main Streamlit App
# ----------------------------------------------------
st.title("Advanced Sudoku Solver with Random Puzzle Generator")

# Sidebar Options
difficulty = st.sidebar.selectbox("Select Puzzle Difficulty", ["Easy", "Medium", "Hard"])
algorithm = st.sidebar.selectbox(
    "Select Solving Method",
    ["Backtracking", "Forward Checking", "CSP (AC3+MRV+Degree+LCV)"]
)
test_runs = st.sidebar.number_input("Number of Test Runs", min_value=1, value=3, step=1)

# Session state to hold the puzzle and its solution.
if "puzzle" not in st.session_state:
    st.session_state["puzzle"] = None
if "solution" not in st.session_state:
    st.session_state["solution"] = None

# Button: Generate a new puzzle.
if st.sidebar.button("Generate Puzzle"):
    with st.spinner("Generating puzzle..."):
        puzzle = generate_puzzle(difficulty)
        st.session_state["puzzle"] = puzzle
        st.session_state["solution"] = None
    st.success("Puzzle generated!")

# Display the generated puzzle.
if st.session_state["puzzle"] is not None:
    display_board_grid(st.session_state["puzzle"], "Generated Puzzle")

# Button: Solve Puzzle with the selected method.
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
            if algorithm == "Backtracking":
                solver.solve_backtracking(board_copy)
            elif algorithm == "Forward Checking":
                domains = solver.init_domains(board_copy)
                solver.solve_forward_checking(board_copy, domains)
            else:  # CSP method with AC3, MRV, Degree, LCV
                domains = solver.init_domains(board_copy)
                solver.solve_csp(board_copy, domains)
            elapsed = time.time() - start_time
            total_time += elapsed
            total_iters += solver.iterations
            solved_board = board_copy
        avg_time = total_time / test_runs
        avg_iters = total_iters / test_runs
        st.write(f"**Solving Method:** {algorithm}")
        st.write(f"**Average Time:** {avg_time:.4f} seconds over {test_runs} runs")
        st.write(f"**Average Iterations:** {avg_iters:.0f}")
        st.session_state["solution"] = solved_board
        display_board_grid(solved_board, "Solved Puzzle")

# Button: Compare performance of all methods on the same puzzle.
if st.sidebar.button("Compare Performance (All Methods)"):
    if st.session_state["puzzle"] is None:
        st.error("Please generate a puzzle first!")
    else:
        puzzle = copy.deepcopy(st.session_state["puzzle"])
        methods = ["Backtracking", "Forward Checking", "CSP (AC3+MRV+Degree+LCV)"]
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
                elif method == "Forward Checking":
                    domains = solver.init_domains(board_copy)
                    solver.solve_forward_checking(board_copy, domains)
                else:
                    domains = solver.init_domains(board_copy)
                    solver.solve_csp(board_copy, domains)
                elapsed = time.time() - start_time
                total_time += elapsed
                total_iters += solver.iterations
            avg_time = total_time / test_runs
            avg_iters = total_iters / test_runs
            results.append((method, avg_time, avg_iters))
        st.subheader("Performance Comparison")
        for res in results:
            st.write(f"**{res[0]}** – Avg Time: {res[1]:.4f}s, Avg Iterations: {res[2]:.0f}")
