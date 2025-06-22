import copy
import numpy as np
import scipy.sparse as sp
from src.core.problem import Problem, ObjectiveSense, ConstraintSense

class SimplexSolver:
    def __init__(self, problem: Problem):
        self.problem = copy.deepcopy(problem)
        self.tableau = None
        self.basis = []
        self.variable_names = []
        self.original_c_std = None

# Dentro da classe SimplexSolver, substitua o método solve por este:

    def solve(self):
        self._add_bounds_as_constraints()
        A_std, b_std, c_std, var_names, slack_basis_candidates, artificial_needed_rows = self._prepare_standard_form()
        
        self.variable_names = var_names
        self.original_c_std = c_std
        
        is_phase1_needed = bool(artificial_needed_rows)

        if not is_phase1_needed:
            self.tableau = self._build_tableau(A_std, b_std, c_std)
            self.basis = slack_basis_candidates
            # Recalcular linha de custo para a base
            for i, basis_var_idx in enumerate(self.basis):
                cost = c_std[basis_var_idx]
                if cost != 0: self.tableau[-1,:] -= cost * self.tableau[i,:]
        else:
            # --- LINHA CORRIGIDA ---
            # Agora estamos passando os argumentos necessários para a função
            success = self._run_phase1(A_std, b_std, slack_basis_candidates, artificial_needed_rows)
            if not success:
                return {"status": "Infeasible"}

        # Fase II: Otimizar
        while True:
            pivot_col = self._find_pivot_column()
            if pivot_col == -1: break
            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row == -1: return {"status": "Unbounded"}
            self._pivot(pivot_row, pivot_col)
            
        return self._extract_solution()

    def _add_bounds_as_constraints(self):
        # Este método está correto e permanece como antes
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

# Dentro da classe SimplexSolver, substitua este método:
    def _prepare_standard_form(self):
        A = self.problem.constraint_matrix.toarray()
        b = self.problem.rhs_vector.copy()
        c = self.problem.objective_coeffs.copy()
        senses = self.problem.constraint_senses.copy()
        var_names = list(self.problem.variable_names)
        
        if self.problem.objective_sense == ObjectiveSense.MAXIMIZE: c = -c
        for i in range(len(b)):
            if b[i] < 0:
                b[i] *= -1; A[i, :] *= -1
                senses[i] = ConstraintSense.GTE if senses[i] == ConstraintSense.LTE else ConstraintSense.LTE

        num_slack_surplus = sum(1 for s in senses if s != ConstraintSense.EQ)
        A_std = np.hstack([A, np.zeros((A.shape[0], num_slack_surplus))])
        c_std = np.hstack([c, np.zeros(num_slack_surplus)])
        
        # Inicializa a lista com um valor placeholder que sabemos ser inválido como índice
        slack_basis_candidates = [-1] * len(senses)
        artificial_needed_rows = []
        slack_surplus_counter = 0
        
        for i, sense in enumerate(senses):
            col_idx = A.shape[1] + slack_surplus_counter
            if sense == ConstraintSense.LTE:
                A_std[i, col_idx] = 1
                var_names.append(f's{i}')
                slack_basis_candidates[i] = col_idx # Guarda o índice da var. de folga
                slack_surplus_counter += 1
            elif sense == ConstraintSense.GTE:
                A_std[i, col_idx] = -1
                var_names.append(f'e{i}')
                artificial_needed_rows.append(i) # Marca a linha para Fase I
                slack_surplus_counter += 1
            else: # '=='
                artificial_needed_rows.append(i) # Marca a linha para Fase I
        
        # --- LINHA CORRIGIDA ---
        return A_std, b, c_std, var_names, slack_basis_candidates, artificial_needed_rows
        
    def _build_initial_tableau(self, A_std, b_std, c_std):
        num_constraints, num_vars = A_std.shape
        basis = [-1] * num_constraints
        needs_phase1 = False
        
        # Encontra a base inicial óbvia (variáveis de folga)
        for i in range(num_constraints):
            one_entries = np.where(A_std[i,:] == 1)[0]
            zero_entries = np.where(A_std[i,:] == 0)[0]
            if len(one_entries) == 1 and len(one_entries) + len(zero_entries) == num_vars:
                col_idx = one_entries[0]
                if np.all(A_std[:, col_idx] == np.eye(num_constraints)[:, i]):
                    basis[i] = col_idx
            
            if basis[i] == -1: needs_phase1 = True

        if not needs_phase1:
            tableau = self._build_tableau(A_std, b_std, c_std)
            # Recalcular linha de custo para a base
            for i, basis_var_idx in enumerate(basis):
                cost = c_std[basis_var_idx]
                if cost != 0: tableau[-1,:] -= cost * tableau[i,:]
            return tableau, basis, False
        else:
            return None, None, True

# Dentro da classe SimplexSolver, substitua este método:
    def _run_phase1(self, A_std, b_std, slack_basis_candidates, artificial_needed_rows):
        print("Fase I necessária...")
        num_constraints, num_vars_std = A_std.shape
        A_phase1 = np.hstack([A_std, np.eye(num_constraints)])
        c_phase1 = np.zeros(A_phase1.shape[1])
        c_phase1[num_vars_std:] = 1
        
        tableau = self._build_tableau(A_phase1, b_std, c_phase1)
        
        # Lógica de inicialização da base robusta
        basis = list(slack_basis_candidates)
        for row_idx in artificial_needed_rows:
            # A variável artificial para a i-ésima restrição que precisa dela,
            # está na coluna correspondente em A_phase1
            basis[row_idx] = num_vars_std + row_idx

        self.tableau, self.basis = tableau, basis
        
        # Ajusta a linha de custo da Fase I
        for row_idx in artificial_needed_rows:
            self.tableau[-1,:] -= self.tableau[row_idx,:]
            
        # Loop do Simplex para a Fase I
        while True:
            pivot_col = self._find_pivot_column()
            if pivot_col == -1: break
            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row == -1: return False # Unbounded na Fase I
            self._pivot(pivot_row, pivot_col)
            
        if self.tableau[-1, -1] > 1e-6:
            print("Problema Inviável (Fase I).")
            return False

        # Prepara a tabela para a Fase II
        tableau_phase2 = self.tableau[:, :num_vars_std+1].copy()
        tableau_phase2[-1, :] = 0
        tableau_phase2[-1, :len(self.original_c_std)] = self.original_c_std
        
        for i, basis_var_idx in enumerate(self.basis):
            if basis_var_idx < len(self.original_c_std):
                cost = self.original_c_std[basis_var_idx]
                if cost != 0:
                    tableau_phase2[-1,:] -= cost * tableau_phase2[i,:]
        
        self.tableau = tableau_phase2
        return True

    def _build_tableau(self, A, b, c):
        num_constraints, num_vars = A.shape
        tableau = np.zeros((num_constraints + 1, num_vars + 1))
        tableau[:num_constraints, :num_vars] = A
        tableau[:num_constraints, -1] = b
        tableau[-1, :num_vars] = c
        return tableau

    def _pivot(self, pivot_row, pivot_col):
        self.tableau[pivot_row, :] /= self.tableau[pivot_row, pivot_col]
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
        if np.all(ratios == np.inf): return -1
        return np.argmin(ratios)

    def _extract_solution(self):
        solution = {name: 0.0 for name in self.problem.variable_names}
        num_orig_vars = len(self.problem.variable_names)
        for i, basis_var_idx in enumerate(self.basis):
            if basis_var_idx < num_orig_vars:
                solution[self.variable_names[basis_var_idx]] = self.tableau[i, -1]
        
        obj_val = 0
        for var, val in solution.items():
            idx = self.problem.variable_names.index(var)
            obj_val += self.problem.objective_coeffs[idx] * val
        return {"status": "Optimal", "variables": solution, "objective_value": obj_val}