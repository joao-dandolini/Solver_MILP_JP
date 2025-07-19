# simplex.py (Versão Final, Puramente Numérica e Correta)

import numpy as np
from typing import List

class SimplexSolver:
    def __init__(self, c: List[float], A: List[List[float]], b: List[float]):
        self.c = np.array(c, dtype=float)
        self.A = np.array(A, dtype=float)
        self.b = np.array(b, dtype=float)
        
        self.num_vars = len(c)
        self.num_constraints = len(b)
        
        self.tableau = None
        self.basic_variables_indices = []


    def add_constraint(self, coeffs: List[float], rhs: float):
        """
        Adiciona uma nova restrição de corte (na forma <=) ao tableau existente.
        """
        num_rows, num_cols = self.tableau.shape
        
        # Cria um novo tableau com uma linha e uma coluna a mais
        new_tableau = np.zeros((num_rows + 1, num_cols + 1))
        
        # Copia os dados existentes
        new_tableau[:num_rows-1, :num_cols-1] = self.tableau[:-1, :-1] # Matriz A e slacks
        new_tableau[:num_rows-1, -1] = self.tableau[:-1, -1]       # RHS
        new_tableau[-1, :] = self.tableau[-1, :]                   # Linha do Objetivo
        
        # Insere a nova restrição na penúltima linha
        new_row = new_tableau[num_rows-1, :]
        new_row[:len(coeffs)] = coeffs
        new_row[-2] = 1.0
        new_row[-1] = rhs
        
        self.tableau = new_tableau
        self.num_constraints += 1
        # Adiciona a nova variável de folga à lista de variáveis básicas
        self.basic_variables_indices.append(num_cols - 1)

    def _initialize_tableau(self):
        num_slack_vars = self.num_constraints
        tableau_cols = self.num_vars + num_slack_vars + 1
        tableau_rows = self.num_constraints + 1
        self.tableau = np.zeros((tableau_rows, tableau_cols))
        
        self.tableau[:self.num_constraints, :self.num_vars] = self.A
        self.tableau[:self.num_constraints, self.num_vars:-1] = np.eye(num_slack_vars)
        self.tableau[:self.num_constraints, -1] = self.b
        self.tableau[-1, :self.num_vars] = -self.c
        
        self.basic_variables_indices = list(range(self.num_vars, self.num_vars + num_slack_vars))

    def _find_pivot_column(self) -> int:
        objective_row = self.tableau[-1, :-1]
        if np.all(objective_row >= -1e-9): return -1
        return np.argmin(objective_row)

    def _find_pivot_row(self, pivot_col_idx: int) -> int:
        min_ratio = float('inf')
        pivot_row = -1
        for i in range(self.num_constraints):
            element = self.tableau[i, pivot_col_idx]
            if element > 1e-9:
                ratio = self.tableau[i, -1] / element
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row = i
        return pivot_row

    def _pivot(self, pivot_row_idx: int, pivot_col_idx: int):
        pivot_element = self.tableau[pivot_row_idx, pivot_col_idx]
        if abs(pivot_element) < 1e-9: return # Evita divisão por zero
        self.tableau[pivot_row_idx, :] /= pivot_element
        for i in range(self.tableau.shape[0]):
            if i != pivot_row_idx:
                multiplier = self.tableau[i, pivot_col_idx]
                self.tableau[i, :] -= multiplier * self.tableau[pivot_row_idx, :]
        self.basic_variables_indices[pivot_row_idx] = pivot_col_idx

    def solve(self) -> 'SimplexSolver':
        self._initialize_tableau()
        max_iterations = (self.num_vars + self.num_constraints) * 5
        for _ in range(max_iterations):
            pivot_col = self._find_pivot_column()
            if pivot_col == -1: break
            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row == -1: return None
            self._pivot(pivot_row, pivot_col)
        return self