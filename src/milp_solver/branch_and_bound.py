import copy
import numpy as np
import scipy.sparse as sp
from src.core.problem import Problem, ConstraintSense
from src.lp_solver.simplex import SimplexSolver

class MILPSolver:
    def __init__(self, problem: Problem):
        """
        Inicializa o solver de Programação Linear Mista-Inteira.
        
        Args:
            problem: O problema original a ser resolvido.
        """
        self.root_problem = problem
        self.node_queue = [] # Nossa "lista de tarefas" de nós a explorar
        
        self.best_integer_solution = None
        self.lower_bound = -float('inf') # Melhor valor de solução inteira (para max)
        self.upper_bound = float('inf') # Melhor valor da relaxação LP

    def _is_integer_feasible(self, solution_vars: dict, problem: Problem) -> bool:
        """
        Verifica se a solução respeita TODAS as restrições de integralidade,
        incluindo as de binariedade.
        """
        for var_idx in problem.integer_variables:
            var_name = problem.variable_names[var_idx]
            value = solution_vars.get(var_name, 0.0)
            
            # 1. Verifica se o valor é fracionário
            if abs(value - round(value)) > 1e-6:
                return False # Não é inteiro

            # 2. Se a variável for binária, verifica se é 0 ou 1
            is_binary = (problem.lower_bounds[var_idx] == 0 and 
                         problem.upper_bounds[var_idx] == 1)
            
            if is_binary and not (abs(value - 0) < 1e-6 or abs(value - 1) < 1e-6):
                return False # É binária, mas o valor não é 0 ou 1
                
        return True

    def _find_branching_variable(self, solution_vars: dict, problem: Problem) -> str:
        """
        Encontra a primeira variável que viola a restrição de integralidade ou binariedade.
        """
        for var_idx in problem.integer_variables:
            var_name = problem.variable_names[var_idx]
            value = solution_vars.get(var_name, 0.0)

            # Verifica se é fracionário
            if abs(value - round(value)) > 1e-6:
                return var_name # Encontrou uma variável fracionária para ramificar

            # Verifica se é uma variável binária com valor inválido
            is_binary = (problem.lower_bounds[var_idx] == 0 and 
                         problem.upper_bounds[var_idx] == 1)

            if is_binary and not (abs(value - 0) < 1e-6 or abs(value - 1) < 1e-6):
                return var_name # Encontrou uma binária com valor ex: 2.0, 100.0 etc.

        return None

    def solve(self):
        print("--- Iniciando o Solver Branch and Bound ---")
        
        # Pré-processa o problema raiz para incluir os bounds
        self.root_problem = self._add_bounds_to_problem_matrix(self.root_problem)
        self.node_queue.append(self.root_problem)

        max_iterations = 500 # Aumentando o limite
        iteration = 0
        while self.node_queue:
            if iteration >= max_iterations:
                print("Número máximo de iterações atingido.")
                break
            
            iteration += 1
            current_problem = self.node_queue.pop()
            
            print(f"\n--- Iteração {iteration}: Resolvendo {current_problem.name} ---")
            lp_solver = SimplexSolver(current_problem)
            solution = lp_solver.solve()
            
            if solution["status"] != "Optimal":
                print("Nó podado (inviável/ilimitado).")
                continue

            if iteration == 1:
                self.upper_bound = solution["objective_value"]
                print(f"Limite Superior Inicial (Upper Bound): {self.upper_bound:.4f}")

            if solution["objective_value"] < self.lower_bound:
                print(f"Nó podado por limite: {solution['objective_value']:.2f} < {self.lower_bound:.2f}")
                continue

            if self._is_integer_feasible(solution["variables"], current_problem):
                print(f"Solução Inteira Encontrada! Valor: {solution['objective_value']:.4f}")
                if solution["objective_value"] > self.lower_bound:
                    self.lower_bound = solution["objective_value"]
                    self.best_integer_solution = solution
                continue

            branch_var_name = self._find_branching_variable(solution["variables"], current_problem)
            if not branch_var_name:
                print("Solução fracionária mas nenhuma variável de ramificação encontrada.")
                continue

            branch_var_value = solution["variables"][branch_var_name]
            var_idx = current_problem.variable_names.index(branch_var_name)
            is_binary = (current_problem.lower_bounds[var_idx] == 0 and current_problem.upper_bounds[var_idx] == 1)
            
            # --- NOVA LÓGICA DE RAMIFICAÇÃO INTELIGENTE ---
            if is_binary:
                print(f"Ramificando na variável BINÁRIA '{branch_var_name}' com valor {branch_var_value:.4f}")
                problem_down = self._add_bound_to_problem(current_problem, branch_var_name, "<=", 0)
                problem_up = self._add_bound_to_problem(current_problem, branch_var_name, ">=", 1)
            else: # Para inteiros gerais
                print(f"Ramificando na variável INTEIRA '{branch_var_name}' com valor {branch_var_value:.4f}")
                problem_down = self._add_bound_to_problem(current_problem, branch_var_name, "<=", np.floor(branch_var_value))
                problem_up = self._add_bound_to_problem(current_problem, branch_var_name, ">=", np.floor(branch_var_value) + 1)
            
            self.node_queue.extend([problem_down, problem_up])

        print("\n--- Fim da Execução do Branch and Bound ---")
        if self.best_integer_solution:
             print("\n--- Melhor Solução Inteira Encontrada ---")
             print(f"Status: {self.best_integer_solution['status']}")
             print(f"Valor da Função Objetivo: {self.best_integer_solution['objective_value']:.4f}")
             print("\nValores das Variáveis:")
             for var, value in self.best_integer_solution["variables"].items():
                 if value > 1e-6:
                     print(f"  {var} = {value:.4f}")
        else:
            print("Nenhuma solução inteira foi encontrada.")
            
        return self.best_integer_solution

    def _add_bound_to_problem(self, problem: Problem, var_name: str, sense_str: str, value: float) -> Problem:
        """Cria uma cópia do problema com uma nova restrição de limite adicionada."""
        new_problem = copy.deepcopy(problem)
        new_problem.name = f"{problem.name}_{var_name}{sense_str}{value}"
        var_idx = new_problem.variable_names.index(var_name)
        
        if sense_str == "<=":
            new_problem.upper_bounds[var_idx] = min(new_problem.upper_bounds[var_idx], value)
        elif sense_str == ">=":
            new_problem.lower_bounds[var_idx] = max(new_problem.lower_bounds[var_idx], value)
            
        return new_problem
    
# Substitua a função inteira pela versão corrigida abaixo
    def _add_bounds_to_problem_matrix(self, problem: Problem):
        """
        Adiciona os limites de variáveis (upper/lower bounds) como restrições
        diretamente na matriz A do problema.
        """
        # Usamos o formato COO por ser eficiente para construir matrizes
        A = problem.constraint_matrix.tocoo()
        b = list(problem.rhs_vector)
        senses = list(problem.constraint_senses)
        
        num_vars = A.shape[1]
        
        # Adiciona restrições de limite inferior (se > 0)
        for i in range(num_vars):
            if problem.lower_bounds[i] > 0 and problem.lower_bounds[i] != -np.inf:
                # CORREÇÃO AQUI: A linha da nossa nova matriz de 1 linha é sempre 0
                row = np.array([0])
                col = np.array([i])
                data = np.array([1])
                new_row_matrix = sp.coo_matrix((data, (row, col)), shape=(1, num_vars))
                A = sp.vstack([A, new_row_matrix])
                b.append(problem.lower_bounds[i])
                senses.append(ConstraintSense.GTE)
                
        # Adiciona restrições de limite superior (se finitos)
        for i in range(num_vars):
            if problem.upper_bounds[i] != np.inf:
                # CORREÇÃO AQUI TAMBÉM
                row = np.array([0])
                col = np.array([i])
                data = np.array([1])
                new_row_matrix = sp.coo_matrix((data, (row, col)), shape=(1, num_vars))
                A = sp.vstack([A, new_row_matrix])
                b.append(problem.upper_bounds[i])
                senses.append(ConstraintSense.LTE)
        
        problem.constraint_matrix = A.tocsr()
        problem.rhs_vector = np.array(b)
        problem.constraint_senses = senses
        return problem