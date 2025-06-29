import copy
import numpy as np
import scipy.sparse as sp
from src.core.problem import Problem, ObjectiveSense, ConstraintSense
import gurobipy as gp
from gurobipy import GRB # Boa prática para ter acesso fácil a constantes como GRB.OPTIMAL

# Dentro da classe Solver, adicione este novo método
def solve_lp_with_gurobi(self, A, b, c, sinais):
    """
    Resolve um problema de programação linear usando Gurobi.
    Retorna status, valor da função objetivo e a solução.
    """
    try:
        # 1. Criar o ambiente e o modelo
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0) # Desliga o log do Gurobi no console
        env.start()
        model = gp.Model("lp_node", env=env)

        # 2. Adicionar variáveis (todas contínuas para a relaxação LP)
        num_vars = A.shape[1]
        x = model.addMVar(shape=num_vars, vtype=GRB.CONTINUOUS, name="x", lb=0) # lb=0 para não-negatividade

        # 3. Definir a função objetivo (assumindo maximização como no seu simplex)
        model.setObjective(c @ x, GRB.MAXIMIZE)

        # 4. Adicionar as restrições
        # É preciso mapear seus 'sinais' para os sentidos do Gurobi
        map_sinais = {'<=': GRB.LESS_EQUAL, '>=': GRB.GREATER_EQUAL, '=': GRB.EQUAL}
        for i in range(A.shape[0]):
            sense = map_sinais[sinais[i]]
            model.addConstr(A[i, :] @ x, sense, b[i], name=f"c{i}")

        # 5. Otimizar o modelo
        model.optimize()

        # 6. Retornar os resultados
        if model.Status == GRB.OPTIMAL:
            return "optimal", model.ObjVal, x.X
        elif model.Status == GRB.INFEASIBLE:
            return "infeasible", None, None
        elif model.Status == GRB.UNBOUNDED:
            return "unbounded", None, None
        else:
            return "error", None, None

    except gp.GurobiError as e:
        print(f"Gurobi error: {e}")
        return "gurobi_error", None, None

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
            # --- MUDANÇA 1: Retorno em caso de falha ---
            if pivot_row == -1: return {"status": "Unbounded"}, None, None
            self._pivot(pivot_row, pivot_col)
            
        # 4. Extrai e valida a solução
        solution = self._extract_solution()
        
        for var_idx in self.basis:
            if self.variable_names[var_idx].startswith('a_'):
                if abs(self.tableau[self.basis.index(var_idx), -1]) > 1e-6:
                       # --- MUDANÇA 2: Retorno em caso de falha ---
                       return {"status": "Infeasible"}, None, None

        # --- MUDANÇA 3: Retorno principal ---
        # Agora retorna a solução, a tabela e a base
        return solution, self.tableau, self.basis

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

# Dentro da classe SimplexSolver, substitua este método:
    def _build_initial_tableau(self, A_std, b_std, c_std, artificial_needed_rows):
        num_constraints, num_vars_std = A_std.shape
        num_artificial = len(artificial_needed_rows)

        A_tableau = np.hstack([A_std, np.zeros((num_constraints, num_artificial))])
        c_tableau = np.hstack([c_std, np.full(num_artificial, self.M)])

        basis = [-1] * num_constraints
        artificial_counter = 0
        
        # Lógica para encontrar a base inicial (slack ou artificial)
        # Primeiro, identificamos as colunas das variáveis de folga
        slack_cols = {}
        slack_surplus_counter = 0
        senses = self.problem.constraint_senses
        for i, sense in enumerate(senses):
             if sense == ConstraintSense.LTE:
                 col_idx = len(self.problem.variable_names) + slack_surplus_counter
                 slack_cols[i] = col_idx
                 slack_surplus_counter += 1
             elif sense == ConstraintSense.GTE:
                 slack_surplus_counter += 1

        # Agora, populamos a base
        for i in range(num_constraints):
            if i in artificial_needed_rows:
                col_idx = num_vars_std + artificial_counter
                A_tableau[i, col_idx] = 1
                self.variable_names.append(f'a_{i}')
                basis[i] = col_idx
                artificial_counter += 1
            else:
                # Se não precisa de artificial, é uma linha de folga
                basis[i] = slack_cols[i]

        tableau = np.zeros((num_constraints + 1, A_tableau.shape[1] + 1))
        tableau[:num_constraints, :A_tableau.shape[1]] = A_tableau
        tableau[:num_constraints, -1] = b_std
        tableau[-1, :len(c_tableau)] = c_tableau
        
        # --- INÍCIO DO BLOCO DE DEBUG ---
        print("\n  DEBUG SIMPLEX: Tabela ANTES do ajuste Big M")
        #print("  Linha de Custo Inicial (c):", tableau[-1, :-1])
        # --------------------------------

        # Ajusta a linha de custo por causa das vars artificiais na base
        for i, basis_idx in enumerate(basis):
            # Se a variável básica tem um custo M, subtraia M * linha da restrição
            if c_tableau[basis_idx] >= self.M:
                print(f"  DEBUG SIMPLEX: Ajustando custo para var artificial na linha {i}")
                tableau[-1,:] -= self.M * tableau[i,:]
        
        # --- INÍCIO DO BLOCO DE DEBUG ---
        print("\n  DEBUG SIMPLEX: Tabela DEPOIS do ajuste Big M")
        #print("  Linha de Custo Ajustada (z-c):", tableau[-1, :-1])
        print("-------------------------------------------------")
        # --------------------------------
        
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
        
        # Pega os valores das variáveis básicas da tabela
        for i, basis_var_idx in enumerate(self.basis):
            if basis_var_idx < num_orig_vars:
                var_name = self.variable_names[basis_var_idx]
                value = self.tableau[i, -1]
                # --- MUDANÇA AQUI: Arredondamos o valor para lidar com a imprecisão ---
                solution[var_name] = round(value, 6)

        # Recalcula o valor da F.O. com os valores já limpos
        obj_val = 0
        for var, val in solution.items():
            if abs(val) > 1e-9:
                idx = self.problem.variable_names.index(var)
                obj_val += self.problem.objective_coeffs[idx] * val
        
        # Arredonda o valor final do objetivo também
        final_objective = round(obj_val, 6)

        return {"status": "Optimal", "variables": solution, "objective_value": final_objective}