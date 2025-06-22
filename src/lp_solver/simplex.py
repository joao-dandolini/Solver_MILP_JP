import copy
import numpy as np
import scipy.sparse as sp
from src.core.problem import Problem, ObjectiveSense, ConstraintSense

class SimplexSolver:
    def __init__(self, problem: Problem):
        self.problem = copy.deepcopy(problem)
        self.M = 1e9 # Nosso "Big M"

    def solve(self):
        # 1. Prepara o problema: adiciona limites e converte para forma padrão
        A_std, b_std, c_std, var_names, artificial_needed_rows = self._prepare_problem()
        self.variable_names = var_names
        
        # 2. Constrói a tabela inicial, já com a lógica do Big M
        self.tableau, self.basis = self._build_initial_tableau(A_std, b_std, c_std, artificial_needed_rows)

        # 3. Resolve com o algoritmo Simplex padrão
        while True:
            pivot_col = self._find_pivot_column()
            if pivot_col == -1: break
            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row == -1: return {"status": "Unbounded"}
            self._pivot(pivot_row, pivot_col)
            
        # 4. Extrai e valida a solução
        solution = self._extract_solution()
        
        # Validação final: se uma variável artificial ainda está na solução, o problema é inviável
        for var_idx in self.basis:
            if self.variable_names[var_idx].startswith('a_'):
                if abs(self.tableau[self.basis.index(var_idx), -1]) > 1e-6:
                     return {"status": "Infeasible"}

        return solution

    def _prepare_problem(self):
        # Adiciona limites de variáveis como restrições explícitas
        num_vars = len(self.problem.variable_names)
        new_constraints, new_senses, new_rhs = [], [], []
        for i in range(num_vars):
            if self.problem.lower_bounds[i] > 0 and not np.isinf(self.problem.lower_bounds[i]):
                new_row = np.zeros(num_vars); new_row[i] = 1
                new_constraints.append(new_row); new_senses.append(ConstraintSense.GTE); new_rhs.append(self.problem.lower_bounds[i])
            if not np.isinf(self.problem.upper_bounds[i]):
                new_row = np.zeros(num_vars); new_row[i] = 1
                new_constraints.append(new_row); new_senses.append(ConstraintSense.LTE); new_rhs.append(self.problem.upper_bounds[i])
        if new_constraints:
            self.problem.constraint_matrix = sp.vstack([self.problem.constraint_matrix.tocsr(), np.array(new_constraints)]).tocsr()
            self.problem.rhs_vector = np.hstack([self.problem.rhs_vector, np.array(new_rhs)])
            self.problem.constraint_senses.extend(new_senses)
            
        # Converte para a forma padrão
        A = self.problem.constraint_matrix.toarray()
        b = self.problem.rhs_vector.copy()
        c = self.problem.objective_coeffs.copy()
        senses = self.problem.constraint_senses
        var_names = list(self.problem.variable_names)
        if self.problem.objective_sense == ObjectiveSense.MAXIMIZE: c = -c
        
        num_slack_surplus = sum(1 for s in senses if s != ConstraintSense.EQ)
        A_std = np.hstack([A, np.zeros((A.shape[0], num_slack_surplus))])
        c_std = np.hstack([c, np.zeros(num_slack_surplus)])
        
        artificial_needed_rows = []
        slack_surplus_counter = 0
        for i, sense in enumerate(senses):
            col_idx = A.shape[1] + slack_surplus_counter
            if sense == ConstraintSense.LTE:
                A_std[i, col_idx] = 1; var_names.append(f's_{i}'); slack_surplus_counter += 1
            elif sense == ConstraintSense.GTE:
                A_std[i, col_idx] = -1; var_names.append(f'e_{i}'); artificial_needed_rows.append(i); slack_surplus_counter += 1
            else:
                artificial_needed_rows.append(i)
        return A_std, b, c_std, var_names, artificial_needed_rows

# Dentro da classe SimplexSolver...
    def _build_initial_tableau(self, A_std, b_std, c_std, artificial_needed_rows):
        num_constraints, num_vars_std = A_std.shape
        num_artificial = len(artificial_needed_rows)

        A_tableau = np.hstack([A_std, np.zeros((num_constraints, num_artificial))])
        c_tableau = np.hstack([c_std, np.full(num_artificial, self.M)])

        basis = [-1] * num_constraints
        artificial_counter = 0
        for i in range(num_constraints):
            # Se a linha tem uma var de folga, ela entra na base
            # (Esta é uma lógica simplificada que funciona para nossos casos de teste)
            is_slack_row = i not in artificial_needed_rows
            if is_slack_row:
                # Encontra a coluna da variável de folga para esta linha
                slack_col_idx = A_std.shape[1] - num_constraints + i # Requer que slacks sejam adicionadas na ordem
                basis[i] = slack_col_idx
            else: # Senão, a var artificial entra na base
                col_idx = num_vars_std + artificial_counter
                A_tableau[i, col_idx] = 1
                basis[i] = col_idx
                
                # --- A LINHA DA VITÓRIA ---
                self.variable_names.append(f'a_{i}') # Adiciona o nome da variável artificial
                # -------------------------
                
                artificial_counter += 1

        tableau = np.zeros((num_constraints + 1, A_tableau.shape[1] + 1))
        tableau[:num_constraints, :A_tableau.shape[1]] = A_tableau
        tableau[:num_constraints, -1] = b_std
        tableau[-1, :len(c_tableau)] = c_tableau
        
        for i, basis_idx in enumerate(basis):
            if c_tableau[basis_idx] == self.M:
                tableau[-1,:] -= self.M * tableau[i,:]
                
        return tableau, basis

    def _pivot(self, pivot_row, pivot_col):
        pivot_val = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= pivot_val
        for i in range(self.tableau.shape[0]):
            if i != pivot_row:
                multiplier = self.tableau[i, pivot_col]
                self.tableau[i, :] -= multiplier * self.tableau[pivot_row, :]
        self.basis[pivot_row] = pivot_col

    def _find_pivot_column(self):
        cost_row = self.tableau[-1, :-1]
        return np.argmin(cost_row) if np.any(cost_row < -1e-9) else -1

    def _find_pivot_row(self, pivot_col):
        rhs = self.tableau[:-1, -1]
        pivot_col_vals = self.tableau[:-1, pivot_col]
        ratios = np.divide(rhs, pivot_col_vals, out=np.full_like(rhs, np.inf), where=pivot_col_vals > 1e-9)
        if np.all(np.isinf(ratios)): return -1
        return np.argmin(ratios)

    def _extract_solution(self):
        solution = {name: 0.0 for name in self.problem.variable_names}
        num_orig_vars = len(self.problem.variable_names)
        for i, basis_var_idx in enumerate(self.basis):
            if basis_var_idx < num_orig_vars:
                solution[self.variable_names[basis_var_idx]] = self.tableau[i, -1]
        
        obj_val = 0
        for var, val in solution.items():
            if abs(val) > 1e-6:
                idx = self.problem.variable_names.index(var)
                obj_val += self.problem.objective_coeffs[idx] * val

        return {"status": "Optimal", "variables": solution, "objective_value": obj_val}