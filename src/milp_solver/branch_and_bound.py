import copy
import numpy as np
import scipy.sparse as sp
import heapq
from src.core.problem import Problem, ConstraintSense
from src.lp_solver.simplex import SimplexSolver

class MILPSolver:
    def __init__(self, problem: Problem):
        self.root_problem = problem
        self.node_queue = []  # Agora é uma fila de prioridade (heap)
        self.best_integer_solution = None
        self.lower_bound = -float('inf')
        self.upper_bound = float('inf')
        self.node_count = 0  # Usado para desempate na fila

    def solve(self):
        print("--- Iniciando o Solver Branch and Bound ---")
        
        # PASSO 1: Resolver o nó raiz uma vez para obter os limites iniciais.
        root_solver = SimplexSolver(self.root_problem)
        root_solution = root_solver.solve()

        if not root_solution or root_solution["status"] != "Optimal":
            print("Problema raiz infactível ou ilimitado. Encerrando.")
            return None
        
        self.upper_bound = root_solution["objective_value"]
        print(f"Limite Superior Inicial (Upper Bound): {self.upper_bound:.4f}")

        # Verifica se a solução do nó raiz já é inteira.
        if self._is_integer_feasible(root_solution["variables"], self.root_problem):
            self.best_integer_solution = root_solution
            self.lower_bound = root_solution["objective_value"]
            print("Solução do nó raiz é inteira e ótima.")
        else:
            # Se não for, adiciona o nó raiz à nossa fila de prioridade para começar a busca.
            self.node_count += 1
            heapq.heappush(self.node_queue, (-root_solution["objective_value"], self.node_count, self.root_problem))

        iteration = 0
        max_iterations = 500

        # PASSO 2: O loop principal do B&B.
        while self.node_queue:
            if iteration >= max_iterations:
                print("Número máximo de iterações B&B atingido.")
                break
            iteration += 1

            # Pega o nó MAIS PROMISSOR da fila (aquele com o maior potencial de lucro).
            lp_bound_neg, _, current_problem = heapq.heappop(self.node_queue)
            
            # Poda por Limite: se o potencial deste nó já é pior que a melhor solução que temos, descarte.
            if -lp_bound_neg < self.lower_bound:
                print(f"Nó podado por limite (Potencial {-lp_bound_neg:.2f} < Melhor achada {self.lower_bound:.2f}).")
                continue

            print(f"\n--- Iteração B&B {iteration}: Resolvendo {current_problem.name} ---")
            
            # Resolve o LP para o problema atual.
            lp_solver = SimplexSolver(current_problem)
            solution = lp_solver.solve()
            
            if not solution or solution["status"] != "Optimal":
                print(f"Nó podado (Status do LP: {solution.get('status', 'Falhou')}).")
                continue

            # Se a solução deste nó for inteira, atualiza nossa melhor solução e poda.
            if self._is_integer_feasible(solution["variables"], current_problem):
                obj_val = self._recalculate_objective(solution["variables"])
                print(f"Solução Inteira Encontrada! Valor: {obj_val:.4f}")
                if obj_val > self.lower_bound:
                    print(f"*** Nova melhor solução encontrada! Lower bound atualizado para {obj_val:.4f} ***")
                    self.lower_bound = obj_val
                    solution['objective_value'] = obj_val
                    self.best_integer_solution = solution
                continue

            # Se a solução for fracionária, ramifica.
            branch_var_name = self._find_branching_variable(solution["variables"], current_problem)
            if not branch_var_name:
                continue

            branch_var_value = solution["variables"][branch_var_name]
            print(f"Ramificando em '{branch_var_name}' com valor {branch_var_value:.4f}")

            # Cria os dois novos ramos (subproblemas).
            branches = [("<=", np.floor(branch_var_value)), (">=", np.floor(branch_var_value) + 1)]
            for sense_str, value in branches:
                new_problem = self._add_bound_to_problem(current_problem, branch_var_name, sense_str, value)
                # Adiciona o novo nó à fila de tarefas. Sua prioridade é o potencial do PAI.
                # A reavaliação será feita quando o nó for retirado da fila.
                self.node_count += 1
                heapq.heappush(self.node_queue, (-solution["objective_value"], self.node_count, new_problem))


        print("\n--- Fim da Execução do Branch and Bound ---")
        if self.best_integer_solution:
             print("\n--- MELHOR SOLUÇÃO INTEIRA ENCONTRADA ---")
             solution_to_print = self.best_integer_solution
             print(f"Status: {solution_to_print['status']}")
             print(f"Valor da Função Objetivo: {solution_to_print['objective_value']:.4f}")
             print("Valores das Variáveis:")
             # Impressão corrigida para mostrar todas as variáveis originais
             for var_name in self.root_problem.variable_names:
                 value = solution_to_print["variables"].get(var_name, 0.0)
                 print(f"  {var_name} = {value:.4f}")
        else:
            print("Nenhuma solução inteira foi encontrada.")
        return self.best_integer_solution

    def _add_bound_to_problem(self, problem: Problem, var_name: str, sense_str: str, value: float) -> Problem:
        new_problem = copy.deepcopy(problem)
        new_problem.name = f"{problem.name}_{var_name}{sense_str}{value}"
        var_idx = new_problem.variable_names.index(var_name)
        if sense_str == "<=":
            new_problem.upper_bounds[var_idx] = min(new_problem.upper_bounds[var_idx], value)
        elif sense_str == ">=":
            new_problem.lower_bounds[var_idx] = max(new_problem.lower_bounds[var_idx], value)
        return new_problem

    def _add_constraint_to_problem(self, problem: Problem, var_name: str, sense_str: str, value: float) -> Problem:
        new_problem = copy.deepcopy(problem)
        new_problem.name = f"{problem.name}_{var_name}{sense_str}{value}"
        var_idx = new_problem.variable_names.index(var_name)
        num_vars = len(new_problem.variable_names)
        new_row = np.zeros((1, num_vars)); new_row[0, var_idx] = 1
        new_problem.constraint_matrix = sp.vstack([new_problem.constraint_matrix.tocsr(), new_row]).tocsr()
        new_problem.rhs_vector = np.append(new_problem.rhs_vector, value)
        new_problem.constraint_senses.append(ConstraintSense(sense_str))
        return new_problem

    def _is_var_binary(self, var_name):
        var_idx = self.root_problem.variable_names.index(var_name)
        return (self.root_problem.lower_bounds[var_idx] == 0 and self.root_problem.upper_bounds[var_idx] == 1)

    def _is_integer_feasible(self, solution_vars: dict, problem: Problem) -> bool:
        # Versão INTELIGENTE que verifica binariedade
        for var_idx in problem.integer_variables:
            var_name = problem.variable_names[var_idx]
            value = solution_vars.get(var_name, 0.0)
            if abs(value - round(value)) > 1e-6: return False
            if self._is_var_binary(var_name) and not (np.isclose(value, 0) or np.isclose(value, 1)): return False
        return True
    
    def _find_branching_variable(self, solution_vars: dict, problem: Problem) -> str:
        # Versão INTELIGENTE que encontra violações de binariedade
        for var_idx in problem.integer_variables:
            var_name = problem.variable_names[var_idx]
            value = solution_vars.get(var_name, 0.0)
            if abs(value - round(value)) > 1e-6: return var_name
            if self._is_var_binary(var_name) and not (np.isclose(value, 0) or np.isclose(value, 1)): return var_name
        return None   
    
    def _recalculate_objective(self, solution_vars):
        obj_val = 0
        for var, val in solution_vars.items():
            if abs(val) > 1e-6:
                idx = self.root_problem.variable_names.index(var)
                obj_val += self.root_problem.objective_coeffs[idx] * val
        return obj_val