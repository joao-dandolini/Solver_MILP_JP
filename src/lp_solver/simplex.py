# Arquivo: src/lp_solver/simplex.py

import numpy as np
import scipy.sparse as sp
from src.core.problem import Problem, ObjectiveSense, ConstraintSense

class SimplexSolver:
    def __init__(self, problem: Problem):
        """Inicializa o solver com o problema a ser resolvido."""
        self.problem = problem
        self.tableau = None
        self.basis = []
        self.variable_names = []

    def _to_standard_form(self):
        """
        Converte um objeto self.problem para a forma padrão para o Simplex.
        """
        A, b, c = self.problem.constraint_matrix.copy(), self.problem.rhs_vector.copy(), self.problem.objective_coeffs.copy()
        
        if self.problem.objective_sense == ObjectiveSense.MAXIMIZE:
            c = -c

        senses = self.problem.constraint_senses.copy()
        for i in range(len(b)):
            if b[i] < 0:
                b[i] = -b[i]
                A[i, :] = -A[i, :]
                if senses[i] == ConstraintSense.LTE:
                    senses[i] = ConstraintSense.GTE
                elif senses[i] == ConstraintSense.GTE:
                    senses[i] = ConstraintSense.LTE

        slack_surplus_vars = []
        
        for i, sense in enumerate(senses):
            if sense == ConstraintSense.EQ:
                continue
            
            # CORREÇÃO APLICADA AQUI
            new_col_data = [-1] if sense == ConstraintSense.GTE else [1]
            
            new_col_indices = [i]
            new_col_indptr = [0, 1]
            
            new_var_col = sp.csc_matrix((new_col_data, new_col_indices, new_col_indptr), shape=(A.shape[0], 1))
            slack_surplus_vars.append(new_var_col)

        if slack_surplus_vars:
            A_std = sp.hstack([A] + slack_surplus_vars, format='csr')
            num_new_vars = len(slack_surplus_vars)
            c_std = np.hstack([c, np.zeros(num_new_vars)])
        else:
            A_std = A
            c_std = c
        
        b_std = b

        print("Conversão para Forma Padrão Concluída:")
        print(f"  - Novas dimensões de A: {A_std.shape}")
        print(f"  - Novo tamanho de c: {len(c_std)}")

        num_new_vars = A_std.shape[1] - len(self.problem.variable_names)
        self.variable_names = self.problem.variable_names + [f's{i}' for i in range(num_new_vars)]
        
        return A_std, b_std, c_std

    def _initialize_tableau(self):
        """
        Monta a tabela Simplex inicial a partir do problema na forma padrão.
        """
        A, b, c = self._to_standard_form()
        A_dense = A.toarray() # A tabela Simplex não costuma ser esparsa

        num_constraints, num_vars = A.shape
        
        # Cria a tabela com espaço para a linha de custo e a coluna do RHS
        tableau = np.zeros((num_constraints + 1, num_vars + 1))
        
        # Preenche a parte das restrições (A e b)
        tableau[:num_constraints, :num_vars] = A_dense
        tableau[:num_constraints, -1] = b
        
        # Preenche a linha de custo (c)
        tableau[-1, :num_vars] = c
        
        self.tableau = tableau
        
        print("Tabela Simplex Inicial Criada:")
        print(self.tableau)

        num_original_vars = len(self.problem.variable_names)
        num_constraints = A.shape[0]
        # As variáveis de folga são as últimas 'num_constraints' colunas
        self.basis = list(range(num_original_vars, num_original_vars + num_constraints))

    
    def _find_pivot_column(self) -> int:
        """
        Encontra o índice da coluna pivô (variável de entrada).
        Retorna o índice da coluna com o custo mais negativo na linha de custo.
        Se não houver negativos, retorna -1 (condição de otimalidade).
        """
        cost_row = self.tableau[-1, :-1]
        if np.all(cost_row >= 0):
            return -1 # Ótimo encontrado
        
        pivot_col = np.argmin(cost_row)
        print(f"Coluna Pivô (Variável de Entrada): {pivot_col}")
        return pivot_col

    def _find_pivot_row(self, pivot_col: int) -> int:
        """
        Encontra o índice da linha pivô (variável de saída) usando o teste da razão mínima.
        """
        rhs_col = self.tableau[:-1, -1]
        pivot_col_data = self.tableau[:-1, pivot_col]
        
        ratios = np.full_like(rhs_col, np.inf) # Inicializa razões como infinito
        
        # Calcula a razão apenas para elementos positivos na coluna pivô
        positive_mask = pivot_col_data > 1e-9 # Usar uma pequena tolerância para evitar divisão por zero
        ratios[positive_mask] = rhs_col[positive_mask] / pivot_col_data[positive_mask]
        
        if np.all(ratios == np.inf):
             # Se todas as razões são infinitas, o problema é ilimitado (unbounded)
            return -1

        pivot_row = np.argmin(ratios)
        print(f"Linha Pivô (Variável de Saída): {pivot_row}")
        return pivot_row

    def _pivot(self, pivot_row: int, pivot_col: int):
        """
        Executa a operação de pivoteamento na tabela.
        """
        pivot_element = self.tableau[pivot_row, pivot_col]
        print(f"\n--- Pivotando no elemento ({pivot_row}, {pivot_col}) com valor {pivot_element:.2f} ---")
        
        # 1. Normalizar a linha pivô
        self.tableau[pivot_row, :] /= pivot_element
        
        # 2. Zerar os outros elementos da coluna pivô
        for i in range(self.tableau.shape[0]):
            if i != pivot_row:
                multiplier = self.tableau[i, pivot_col]
                self.tableau[i, :] -= multiplier * self.tableau[pivot_row, :]
        
        self.basis[pivot_row] = pivot_col
        print("Tabela após o pivoteamento:")
        print(self.tableau)

    def _extract_solution(self):
        """
        Extrai a solução final da tabela ótima.
        """
        solution = {name: 0.0 for name in self.problem.variable_names}
        objective_value = 0
        
        # O valor da F.O. está no canto inferior direito
        # Lembre-se que minimizamos -c*x para um problema de maximização
        final_obj_val = self.tableau[-1, -1]
        if self.problem.objective_sense == ObjectiveSense.MAXIMIZE:
            objective_value = final_obj_val
        else:
            objective_value = -final_obj_val
            
        # Pega os valores das variáveis básicas
        rhs_col = self.tableau[:-1, -1]
        for i, basis_var_idx in enumerate(self.basis):
            # Só reportamos as variáveis originais do problema
            if basis_var_idx < len(self.problem.variable_names):
                var_name = self.variable_names[basis_var_idx]
                solution[var_name] = rhs_col[i]

        return {
            "status": "Optimal",
            "variables": solution,
            "objective_value": objective_value
        }    

    def solve(self):
        self._initialize_tableau()
        
        max_iterations = 100
        for iteration in range(max_iterations):
            pivot_col = self._find_pivot_column()
            if pivot_col == -1:
                print("\nCondição de otimalidade atingida.")
                return self._extract_solution() # <<<<<<< NOVO: Retorna a solução formatada

            pivot_row = self._find_pivot_row(pivot_col)
            if pivot_row == -1:
                print("Problema é ilimitado (unbounded).")
                return {"status": "Unbounded"}
                
            self._pivot(pivot_row, pivot_col)
        
        print("Número máximo de iterações atingido.")
        return {"status": "Max Iterations Reached"}