import streamlit as st
import time
import random
import copy
import numpy as np
import pandas as pd
from typing import List, Tuple, Set, Dict
from collections import deque

class SudokuCSP:
    def __init__(self, board: List[List[int]]):
        self.board = board
        self.variables = [(i, j) for i in range(9) for j in range(9)]
        self.domains = {(i, j): {board[i][j]} if board[i][j] != 0 else set(range(1, 10)) 
                        for i, j in self.variables}
        self.neighbors = self._build_neighbors()
        self.iterations = 0

    def _build_neighbors(self) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
        neighbors = {(i, j): set() for i, j in self.variables}
        for i in range(9):
            for j in range(9):
                for jj in range(9):
                    if jj != j:
                        neighbors[(i, j)].add((i, jj))
                for ii in range(9):
                    if ii != i:
                        neighbors[(i, j)].add((ii, j))
                box_i, box_j = 3 * (i // 3), 3 * (j // 3)
                for ii in range(box_i, box_i + 3):
                    for jj in range(box_j, box_j + 3):
                        if (ii, jj) != (i, j):
                            neighbors[(i, j)].add((ii, jj))
        return neighbors

    def is_consistent(self, var: Tuple[int, int], value: int) -> bool:
        i, j = var
        for ni, nj in self.neighbors[var]:
            if self.board[ni][nj] == value:
                return False
        return True

    @staticmethod
    def is_valid(board: List[List[int]], row: int, col: int, num: int) -> bool:
        for x in range(9):
            if board[row][x] == num or board[x][col] == num:
                return False
        box_i, box_j = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_i, box_i + 3):
            for j in range(box_j, box_j + 3):
                if board[i][j] == num:
                    return False
        return True

    @staticmethod
    def generate_random_sudoku(clues: int = 30) -> List[List[int]]:
        board = [[0 for _ in range(9)] for _ in range(9)]
        
        def fill_board(board):
            for i in range(9):
                for j in range(9):
                    if board[i][j] == 0:
                        values = list(range(1, 10))
                        random.shuffle(values)
                        for value in values:
                            if SudokuCSP.is_valid(board, i, j, value):
                                board[i][j] = value
                                if fill_board(board):
                                    return True
                                board[i][j] = 0
                        return False
            return True
        
        fill_board(board)
        
        cells = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(cells)
        remove_count = 81 - clues
        for i in range(min(remove_count, len(cells))):
            row, col = cells[i]
            board[row][col] = 0
        
        return board

    def backtracking_search(self) -> bool:
        self.iterations = 0
        def backtrack():
            self.iterations += 1
            unassigned = [(i, j) for i, j in self.variables if self.board[i][j] == 0]
            if not unassigned:
                return True
            var = unassigned[0]
            i, j = var
            for value in self.domains[var]:
                if self.is_consistent(var, value):
                    self.board[i][j] = value
                    if backtrack():
                        return True
                    self.board[i][j] = 0
            return False
        return backtrack()

    def forward_checking(self, var: Tuple[int, int], value: int, domains: Dict[Tuple[int, int], Set[int]]) -> bool:
        i, j = var
        for ni, nj in self.neighbors[var]:
            if (ni, nj) in domains and value in domains[(ni, nj)]:
                domains[(ni, nj)].remove(value)
                if not domains[(ni, nj)]:
                    return False
        return True

    def revise(self, var1: Tuple[int, int], var2: Tuple[int, int], domains: Dict[Tuple[int, int], Set[int]]) -> bool:
        revised = False
        values = domains[var1].copy()
        for x in values:
            if not any(x != y for y in domains[var2]):
                domains[var1].remove(x)
                revised = True
        return revised

    def ac3(self, domains: Dict[Tuple[int, int], Set[int]]) -> bool:
        queue = deque([(v1, v2) for v1 in self.variables for v2 in self.neighbors[v1]])
        while queue:
            (xi, xj) = queue.popleft()
            if self.revise(xi, xj, domains):
                if not domains[xi]:
                    return False
                for xk in self.neighbors[xi]:
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    def backtracking_fc(self) -> bool:
        self.iterations = 0
        def backtrack(domains):
            self.iterations += 1
            unassigned = [(i, j) for i, j in self.variables if self.board[i][j] == 0]
            if not unassigned:
                return True
            var = unassigned[0]
            i, j = var
            for value in domains[var].copy():
                if self.is_consistent(var, value):
                    self.board[i][j] = value
                    saved_domains = {v: d.copy() for v, d in domains.items()}
                    if self.forward_checking(var, value, domains):
                        if backtrack(domains):
                            return True
                    domains.update(saved_domains)
                    self.board[i][j] = 0
            return False
        domains = {v: d.copy() for v, d in self.domains.items()}
        return backtrack(domains)

    def backtracking_ac3(self) -> bool:
        self.iterations = 0
        domains = {v: d.copy() for v, d in self.domains.items()}
        if not self.ac3(domains):
            return False
        def backtrack(domains):
            self.iterations += 1
            unassigned = [(i, j) for i, j in self.variables if self.board[i][j] == 0]
            if not unassigned:
                return True
            var = unassigned[0]
            i, j = var
            for value in domains[var].copy():
                if self.is_consistent(var, value):
                    self.board[i][j] = value
                    saved_domains = {v: d.copy() for v, d in domains.items()}
                    domains[var] = {value}
                    if self.ac3(domains):
                        if backtrack(domains):
                            return True
                    domains.update(saved_domains)
                    self.board[i][j] = 0
            return False
        return backtrack(domains)

    def select_unassigned_variable(self, domains: Dict[Tuple[int, int], Set[int]], heuristic: str = 'none') -> Tuple[int, int]:
        unassigned = [(i, j) for i, j in self.variables if self.board[i][j] == 0]
        if not unassigned:
            return None
        if heuristic == 'mrv':
            return min(unassigned, key=lambda var: len(domains[var]))
        elif heuristic == 'degree':
            return max(unassigned, key=lambda var: sum(1 for n in self.neighbors[var] if self.board[n[0]][n[1]] == 0))
        elif heuristic == 'mrv+degree':
            min_mrv = min(len(domains[var]) for var in unassigned)
            candidates = [var for var in unassigned if len(domains[var]) == min_mrv]
            return max(candidates, key=lambda var: sum(1 for n in self.neighbors[var] if self.board[n[0]][n[1]] == 0))
        return unassigned[0]

    def order_domain_values(self, var: Tuple[int, int], domains: Dict[Tuple[int, int], Set[int]], heuristic: str = 'none') -> List[int]:
        if heuristic == 'lcv':
            def count_conflicts(value):
                count = 0
                for ni, nj in self.neighbors[var]:
                    if (ni, nj) in domains and value in domains[(ni, nj)]:
                        count += 1
                return count
            return sorted(domains[var], key=count_conflicts)
        return list(domains[var])

    def backtracking_with_heuristics(self, heuristic_var: str = 'none', heuristic_val: str = 'none') -> bool:
        self.iterations = 0
        def backtrack(domains):
            self.iterations += 1
            var = self.select_unassigned_variable(domains, heuristic_var)
            if not var:
                return True
            i, j = var
            for value in self.order_domain_values(var, domains, heuristic_val):
                if self.is_consistent(var, value):
                    self.board[i][j] = value
                    saved_domains = {v: d.copy() for v, d in domains.items()}
                    if self.forward_checking(var, value, domains):
                        if backtrack(domains):
                            return True
                    domains.update(saved_domains)
                    self.board[i][j] = 0
            return False
        domains = {v: d.copy() for v, d in self.domains.items()}
        return backtrack(domains)

def display_grid(board, title, container=None):
    try:
        # Use container if provided, else use st directly
        target = container if container is not None else st
        with target.container():
            st.subheader(title)
            board_np = np.array(board, dtype=str)
            board_np[board_np == '0'] = ''
            df = pd.DataFrame(board_np)
            styled_df = df.style.set_properties(**{
                'text-align': 'center',
                'font-size': '20px',
                'border': '1px solid black',
                'width': '50px',
                'height': '50px'
            })
            for i in range(9):
                for j in range(9):
                    borders = {}
                    if i % 3 == 0 and i > 0:
                        borders['border-top'] = '3px solid black'
                    if j % 3 == 0 and j > 0:
                        borders['border-left'] = '3px solid black'
                    if borders:
                        styled_df = styled_df.set_properties(**borders, subset=pd.IndexSlice[i, j])
            st.dataframe(styled_df, use_container_width=False)
            # Fallback: Display raw board as text
            st.text("Raw board content:")
            for row in board:
                st.text(str(row))
    except Exception as e:
        st.error(f"Error displaying grid: {str(e)}")
        st.text("Raw board content (fallback):")
        for row in board:
            st.text(str(row))

def compute_metrics(board):
    methods = {
        "Basic Backtracking": lambda s: s.backtracking_search(),
        "Forward Checking": lambda s: s.backtracking_fc(),
        "Arc Consistency": lambda s: s.backtracking_ac3(),
        "Heuristics (MRV + Degree + LCV)": lambda s: s.backtracking_with_heuristics("mrv+degree", "lcv")
    }
    metrics = {}
    for method_name, solver in methods.items():
        times = []
        iterations = 0
        for _ in range(5):  # Reduced runs to lower resource usage
            sudoku = SudokuCSP(copy.deepcopy(board))
            start_time = time.time()
            if not solver(sudoku):
                return None
            end_time = time.time()
            times.append(end_time - start_time)
            if _ == 0:
                iterations = sudoku.iterations
        avg_time = sum(times) / len(times)
        metrics[method_name] = {"iterations": iterations, "avg_time": avg_time}
    return metrics

def main():
    st.title("Sudoku Solver")
    st.write("Debug: App initialized.")
    
    # Initialize session state
    if 'board' not in st.session_state:
        try:
            st.session_state.board = SudokuCSP.generate_random_sudoku(clues=30)
            st.session_state.original_board = copy.deepcopy(st.session_state.board)
            st.session_state.solved_board = None
            st.session_state.performance = None
            st.session_state.metrics = compute_metrics(st.session_state.board)
            st.session_state.difficulty = "Medium"
            st.write("Debug: Session state initialized.")
        except Exception as e:
            st.error(f"Error initializing session state: {str(e)}")
            return

    # Display initial puzzle
    try:
        display_grid(st.session_state.board, "Initial Puzzle")
        st.write("Debug: Initial puzzle displayed.")
    except Exception as e:
        st.error(f"Error displaying initial puzzle: {str(e)}")

    # Solved puzzle container
    solved_container = st.container()

    # Controls
    st.subheader("Controls")
    try:
        difficulty = st.selectbox("Select Difficulty:", ["Easy", "Medium", "Hard"], 
                                 index=["Easy", "Medium", "Hard"].index(st.session_state.difficulty))
        clue_map = {"Easy": 40, "Medium": 30, "Hard": 25}
        
        if difficulty != st.session_state.difficulty:
            st.session_state.difficulty = difficulty
            st.session_state.board = SudokuCSP.generate_random_sudoku(clues=clue_map[difficulty])
            st.session_state.original_board = copy.deepcopy(st.session_state.board)
            st.session_state.solved_board = None
            st.session_state.performance = None
            st.session_state.metrics = compute_metrics(st.session_state.board)
            solved_container.empty()
            st.write("Debug: Difficulty changed and puzzle updated.")
    except Exception as e:
        st.error(f"Error handling difficulty selection: {str(e)}")

    try:
        method = st.selectbox("Select Solving Method:", 
                             ["Basic Backtracking", "Forward Checking", "Arc Consistency", "Heuristics (MRV + Degree + LCV)"])
        
        method_map = {
            "Basic Backtracking": "basic",
            "Forward Checking": "fc",
            "Arc Consistency": "ac3",
            "Heuristics (MRV + Degree + LCV)": "heuristics"
        }
        selected_method = method_map[method]
        st.write("Debug: Method selected.")
    except Exception as e:
        st.error(f"Error handling method selection: {str(e)}")

    # Solve button
    if st.button("Solve"):
        with solved_container:
            st.write("Debug: Starting solve process...")
            try:
                sudoku = SudokuCSP(copy.deepcopy(st.session_state.board))
                times = []
                
                if selected_method == "basic":
                    solver = sudoku.backtracking_search
                elif selected_method == "fc":
                    solver = sudoku.backtracking_fc
                elif selected_method == "ac3":
                    solver = sudoku.backtracking_ac3
                else:
                    solver = lambda: sudoku.backtracking_with_heuristics("mrv+degree", "lcv")

                for _ in range(5):  # Reduced runs to lower resource usage
                    temp_sudoku = SudokuCSP(copy.deepcopy(st.session_state.board))
                    start_time = time.time()
                    if not solver():
                        st.error(f"No solution exists for {method}!")
                        st.write("Debug: Solver returned False.")
                        st.session_state.solved_board = None
                        return
                    end_time = time.time()
                    times.append(end_time - start_time)
                    if _ == 0:
                        st.session_state.solved_board = temp_sudoku.board

                avg_time = sum(times) / len(times)
                st.session_state.performance = f"Selected Method Performance ({method}): {avg_time:.4f} seconds (avg over 5 runs)"
                st.write("Debug: Solver completed. Solved board set.")
                if st.session_state.solved_board is not None:
                    st.write("Debug: Displaying solved board...")
                    display_grid(st.session_state.solved_board, "Solved Puzzle", solved_container)
                    st.write(st.session_state.performance)
                else:
                    st.write("Debug: Solved board is None after solving.")
            except Exception as e:
                st.error(f"Error during solving: {str(e)}")

    # Display solved puzzle if already set
    if st.session_state.solved_board is not None:
        with solved_container:
            st.write("Debug: Rendering existing solved board...")
            try:
                display_grid(st.session_state.solved_board, "Solved Puzzle", solved_container)
                if st.session_state.performance:
                    st.write(st.session_state.performance)
            except Exception as e:
                st.error(f"Error rendering solved puzzle: {str(e)}")

    # New puzzle button
    if st.button("New Puzzle"):
        try:
            st.session_state.board = SudokuCSP.generate_random_sudoku(clues=clue_map[st.session_state.difficulty])
            st.session_state.original_board = copy.deepcopy(st.session_state.board)
            st.session_state.solved_board = None
            st.session_state.performance = None
            st.session_state.metrics = compute_metrics(st.session_state.board)
            solved_container.empty()
            st.write("Debug: New puzzle generated.")
        except Exception as e:
            st.error(f"Error generating new puzzle: {str(e)}")

    # Comparison table
    st.subheader("Comparison of Solving Methods")
    try:
        if st.session_state.metrics:
            metrics_data = []
            for method_name, data in st.session_state.metrics.items():
                metrics_data.append({
                    "Method": method_name,
                    "Iterations": data["iterations"],
                    "Avg Time (seconds)": f"{data['avg_time']:.4f}"
                })
            df = pd.DataFrame(metrics_data)
            styled_df = df.style.set_properties(**{
                'text-align': 'center',
                'font-size': '14px',
                'border': '1px solid black'
            }).set_table_styles([
                {'selector': 'th', 'props': [('font-weight', 'bold'), ('background-color', '#f0f0f0'), ('border', '1px solid black')]}
            ])
            st.dataframe(styled_df, use_container_width=True)
            st.write("Debug: Comparison table displayed.")
        else:
            st.error("Unable to compute metrics for this puzzle.")
    except Exception as e:
        st.error(f"Error displaying comparison table: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"App failed to start: {str(e)}")
        st.write("Debug: App crashed. Check logs for details.")
