import streamlit as st
import time
import copy
import random
import pandas as pd

# ----------------------------------------------------
# Utility: Check if placing 'num' at (row, col) is valid.
# ----------------------------------------------------
def is_valid(board, row, col, num):
    # Row check.
    for c in range(9):
        if board[row][c] == num:
            return False
    # Column check.
    for r in range(9):
        if board[r][col] == num:
            return False
    # Check 3x3 sub-grid.
    start_row, start_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(start_row, start_row + 3):
        for c in range(start_col, start_col + 3):
            if board[r][c] == num:
                return False
    return True

# ----------------------------------------------------
# SudokuSolver: Contains various backtracking solver variants.
# ----------------------------------------------------
class SudokuSolver:
    def __init__(self, board):
        self.board = board
        self.iterations = 0  # Count of recursive calls

    def find_empty(self, board):
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    return (r, c)
        return None

    # --------------------------
    # 1. Basic Backtracking (BT)
    # --------------------------
    def solve_backtracking(self, board):
        self.iterations += 1
        empty = self.find_empty(board)
        if not empty:
            return True
        row, col = empty
        for num in range(1, 10):
            if is_valid(board, row, col, num):
                board[row][col] = num
                if self.solve_backtracking(board):
                    return True
                board[row][col] = 0  # Backtrack
        return False

    # ------------------------------------------------
    # 2. BT with Forward Checking (FC)
    # ------------------------------------------------
    def solve_forward_checking(self, board, domains):
        self.iterations += 1
        # Find an empty cell—if none, puzzle is solved.
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
        # Use MRV: choose the empty cell with the smallest domain.
        min_size = 10
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0 and len(domains[(r, c)]) < min_size:
                    min_size = len(domains[(r, c)])
                    row, col = r, c
        # Try each candidate from the cell's domain.
        for num in sorted(domains[(row, col)]):
            if is_valid(board, row, col, num):
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

    # ------------------------------------------------
    # 3. BT with Arc Consistency (AC3 Filtering)
    # ------------------------------------------------
    def get_neighbors(self, row, col):
        neighbors = set()
        for i in range(9):
            if i != col:
                neighbors.add((row, i))
            if i != row:
                neighbors.add((i, col))
        sr, sc = 3 * (row // 3), 3 * (col // 3)
        for r in range(sr, sr + 3):
            for c in range(sc, sc + 3):
                if (r, c) != (row, col):
                    neighbors.add((r, c))
        return neighbors

    def init_domains(self, board):
        domains = {}
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    domains[(r, c)] = set(range(1, 10))
                else:
                    domains[(r, c)] = {board[r][c]}
        for r in range(9):
            for c in range(9):
                if board[r][c] != 0:
                    num = board[r][c]
                    for neighbor in self.get_neighbors(r, c):
                        if board[neighbor[0]][neighbor[1]] == 0 and num in domains[neighbor]:
                            domains[neighbor].remove(num)
        return domains

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
        # Under the "≠" constraint, if every value in domain[xj] equals a, then remove a.
        for a in set(domains[xi]):
            if all(b == a for b in domains[xj]):
                domains[xi].remove(a)
                revised = True
        return revised

    def solve_backtracking_ac3(self, board, domains):
        self.iterations += 1
        domains = self.ac3(domains)
        if domains is False:
            return False
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
        cell = None
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    cell = (r, c)
                    break
            if cell:
                break
        row, col = cell
        for num in sorted(domains[(row, col)]):
            if is_valid(board, row, col, num):
                board[row][col] = num
                new_domains = copy.deepcopy(domains)
                new_domains[(row, col)] = {num}
                if self.solve_backtracking_ac3(board, new_domains):
                    return True
                board[row][col] = 0
        return False

    # --------------------------
    # 4. BT with MRV (Minimum Remaining Values)
    # --------------------------
    def find_empty_mrv(self, board):
        best = None
        min_options = 10
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    options = sum(1 for num in range(1, 10) if is_valid(board, r, c, num))
                    if options < min_options:
                        min_options = options
                        best = (r, c)
        return best

    def solve_backtracking_mrv(self, board):
        self.iterations += 1
        cell = self.find_empty_mrv(board)
        if not cell:
            return True
        row, col = cell
        for num in range(1, 10):
            if is_valid(board, row, col, num):
                board[row][col] = num
                if self.solve_backtracking_mrv(board):
                    return True
                board[row][col] = 0
        return False

    # --------------------------
    # 5. BT with MRV + Degree Heuristic
    # --------------------------
    def find_empty_mrv_degree(self, board):
        best = None
        min_options = 10
        best_degree = -1
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    options = sum(1 for num in range(1, 10) if is_valid(board, r, c, num))
                    degree = sum(1 for (nr, nc) in self.get_neighbors(r, c) if board[nr][nc] == 0)
                    if options < min_options or (options == min_options and degree > best_degree):
                        best = (r, c)
                        min_options = options
                        best_degree = degree
        return best

    def solve_backtracking_mrv_degree(self, board):
        self.iterations += 1
        cell = self.find_empty_mrv_degree(board)
        if not cell:
            return True
        row, col = cell
        for num in range(1, 10):
            if is_valid(board, row, col, num):
                board[row][col] = num
                if self.solve_backtracking_mrv_degree(board):
                    return True
                board[row][col] = 0
        return False

    # --------------------------
    # 6. BT with LCV (Least Constraining Value)
    # --------------------------
    def order_values_lcv(self, board, row, col):
        candidates = [num for num in range(1, 10) if is_valid(board, row, col, num)]
        def impact(num):
            impact_count = 0
            board[row][col] = num
            for (nr, nc) in self.get_neighbors(row, col):
                if board[nr][nc] == 0:
                    impact_count += sum(1 for k in range(1, 10) if is_valid(board, nr, nc, k))
            board[row][col] = 0
            return impact_count
        candidates.sort(key=impact, reverse=True)
        return candidates

    def solve_backtracking_lcv(self, board):
        self.iterations += 1
        empty = self.find_empty(board)
        if not empty:
            return True
        row, col = empty
        for num in self.order_values_lcv(board, row, col):
            if is_valid(board, row, col, num):
                board[row][col] = num
                if self.solve_backtracking_lcv(board):
                    return True
                board[row][col] = 0
        return False

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
    # Use a randomized BT to complete the board.
    solver.solve_backtracking(board)
    return board

def generate_puzzle(difficulty):
    complete_board = generate_complete_board()
    # Define the number of clues based on difficulty.
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
    df = pd.DataFrame(board, columns=[str(i+1) for i in range(9)])
    st.table(df)

# ----------------------------------------------------
# Main Streamlit App
# ----------------------------------------------------
st.title("Sudoku Solver – Backtracking Variants & Performance Analysis")

# Sidebar: Puzzle Difficulty and Method Selection.
difficulty = st.sidebar.selectbox("Select Puzzle Difficulty", ["Easy", "Medium", "Hard"])
method_selection = st.sidebar.multiselect(
    "Select Solving Method(s) (select one or more)",
    options=[
        "Basic Backtracking",
        "BT with Forward Checking",
        "BT with Arc Consistency",
        "BT with MRV",
        "BT with MRV + Degree",
        "BT with LCV"
    ],
    default=["Basic Backtracking"]
)
test_runs = st.sidebar.number_input("Number of Test Runs", min_value=1, value=3, step=1)

# Button to generate a new puzzle.
if st.sidebar.button("Generate Puzzle"):
    with st.spinner("Generating puzzle..."):
        puzzle = generate_puzzle(difficulty)
        st.session_state["puzzle"] = puzzle
        st.session_state["solution"] = None
    st.success("Puzzle generated!")

# Display the generated puzzle.
if "puzzle" in st.session_state and st.session_state["puzzle"] is not None:
    display_board_grid(st.session_state["puzzle"], "Generated Puzzle")

# run_solver: Runs the selected method on a copy of the puzzle.
def run_solver(method, puzzle):
    board_copy = copy.deepcopy(puzzle)
    solver = SudokuSolver(board_copy)
    start_time = time.time()
    if method == "Basic Backtracking":
        solver.solve_backtracking(board_copy)
    elif method == "BT with Forward Checking":
        domains = solver.init_domains(board_copy)
        solver.solve_forward_checking(board_copy, domains)
    elif method == "BT with Arc Consistency":
        domains = solver.init_domains(board_copy)
        solver.solve_backtracking_ac3(board_copy, domains)
    elif method == "BT with MRV":
        solver.solve_backtracking_mrv(board_copy)
    elif method == "BT with MRV + Degree":
        solver.solve_backtracking_mrv_degree(board_copy)
    elif method == "BT with LCV":
        solver.solve_backtracking_lcv(board_copy)
    elapsed = time.time() - start_time
    return board_copy, elapsed, solver.iterations

# Button: Solve Puzzle using the selected method(s).
if st.sidebar.button("Solve Puzzle"):
    if "puzzle" not in st.session_state or st.session_state["puzzle"] is None:
        st.error("Please generate a puzzle first!")
    else:
        puzzle = st.session_state["puzzle"]
        results = []
        for method in method_selection:
            total_time = 0
            total_iters = 0
            solved_board = None
            for _ in range(test_runs):
                board_copy, elapsed, iters = run_solver(method, puzzle)
                total_time += elapsed
                total_iters += iters
                solved_board = board_copy
            avg_time = total_time / test_runs
            avg_iters = total_iters / test_runs
            results.append({"Method": method,
                            "Avg Time (s)": round(avg_time, 4),
                            "Avg Iterations": int(avg_iters)})
            if len(method_selection) == 1:
                st.session_state["solution"] = solved_board

        # If only one method is selected, display the solved puzzle.
        if len(method_selection) == 1 and st.session_state.get("solution") is not None:
            display_board_grid(st.session_state["solution"], "Solved Puzzle")
        
        st.subheader("Performance Metrics")
        df_results = pd.DataFrame(results)
        st.table(df_results)

# Optional: Button to compare performance of all methods.
if st.sidebar.button("Compare Performance (All Methods)"):
    if "puzzle" not in st.session_state or st.session_state["puzzle"] is None:
        st.error("Please generate a puzzle first!")
    else:
        puzzle = st.session_state["puzzle"]
        all_methods = [
            "Basic Backtracking",
            "BT with Forward Checking",
            "BT with Arc Consistency",
            "BT with MRV",
            "BT with MRV + Degree",
            "BT with LCV"
        ]
        results = []
        for method in all_methods:
            total_time = 0
            total_iters = 0
            for _ in range(test_runs):
                _, elapsed, iters = run_solver(method, puzzle)
                total_time += elapsed
                total_iters += iters
            avg_time = total_time / test_runs
            avg_iters = total_iters / test_runs
            results.append({"Method": method,
                            "Avg Time (s)": round(avg_time, 4),
                            "Avg Iterations": int(avg_iters)})
        st.subheader("Performance Comparison (All Methods)")
        df_all = pd.DataFrame(results)
        st.table(df_all)
