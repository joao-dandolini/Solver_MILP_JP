import copy
import numpy as np
import scipy.sparse as sp
import heapq
from src.core.problem import Problem, ConstraintSense
from src.lp_solver.simplex import SimplexSolver
from.cuts import generate_gomory_cut

TOLERANCE = 1e-6

class MILPSolver:
    def __init__(self, problem: Problem):
        self.root_problem = problem
        self.node_queue = []  # Agora é uma fila de prioridade (heap)
        self.best_integer_solution = None
        self.lower_bound = -float('inf')
        self.upper_bound = float('inf')
        self.node_count = 0  # Usado para desempate na fila

# Dentro da classe MILPSolver, substitua o método solve inteiro por este:
    def solve(self):
        print("--- Iniciando o Solver Branch and Bound ---")
        self.node_queue.append(self.root_problem)

        iteration = 0
        max_b_and_b_iterations = 200

        while self.node_queue:
            if iteration >= max_b_and_b_iterations:
                print("Número máximo de iterações B&B atingido.")
                break
            iteration += 1

            current_problem = self.node_queue.pop(0)
            print(f"\n--- Iteração B&B {iteration}: Resolvendo {current_problem.name} ---")

            # --- INÍCIO DO LOOP DE CORTE ---
            # Para cada nó, tentamos adicionar até 10 cortes para fortalecê-lo
            max_cuts_per_node = 10
            for cut_iteration in range(max_cuts_per_node):
                print(f"  Tentativa de corte {cut_iteration + 1}/{max_cuts_per_node}...")
                
                lp_solver = SimplexSolver(current_problem)
                solution, final_tableau, final_basis = lp_solver.solve()
                
                # Se o LP falhar, não podemos gerar cortes, saia do loop de corte
                if not solution or solution["status"] != "Optimal":
                    print("  LP do nó falhou, impossível gerar cortes.")
                    break

                # Se a solução já for inteira, não precisamos de cortes
                if self._is_integer_feasible(solution["variables"], current_problem):
                    print("  Solução tornou-se inteira após corte. Saindo do loop de corte.")
                    break

                # Tenta gerar um corte de Gomory
                new_cut = generate_gomory_cut(final_tableau, final_basis, lp_solver.variable_names, current_problem)
                
                if new_cut:
                    #'''
                    cut_coeffs = new_cut["coeffs"]
                    cut_rhs = new_cut["rhs"]
                    cut_sense = ">="
                    # Formata a equação do corte para ser legível
                    lhs_str = " + ".join([f"{coeff:.3f}*{var}" for var, coeff in cut_coeffs.items()])
                    print(f"  --> Corte de Gomory Gerado: {lhs_str} {cut_sense} {cut_rhs:.3f}")
                    #'''
                    print("  Corte de Gomory encontrado! Adicionando ao problema e re-otimizando...")
                    # O problema é atualizado com o novo corte para a próxima iteração do loop de corte
                    current_problem = self._add_constraint_to_problem(
                        current_problem, new_cut["coeffs"], new_cut["sense"], new_cut["rhs"]
                    )
                else:
                    # Se não há mais cortes a adicionar, saia do loop de corte
                    print("  Nenhum corte de Gomory adicional encontrado.")
                    break
            # --- FIM DO LOOP DE CORTE ---

            # --- ANÁLISE DO RESULTADO FINAL DO NÓ (APÓS TENTATIVAS DE CORTE) ---
            if not solution or solution["status"] != "Optimal":
                print(f"Nó podado. Status final do LP: {solution.get('status', 'Falhou')}")
                continue

            if iteration == 1:
                self.upper_bound = solution["objective_value"]
                print(f"Limite Superior Inicial (Upper Bound): {self.upper_bound:.4f}")

            if solution["objective_value"] < self.lower_bound:
                print(f"Nó podado por limite: LP bound {solution['objective_value']:.2f} < Best integer {self.lower_bound:.2f}")
                continue

            if self._is_integer_feasible(solution["variables"], current_problem):
                obj_val = self._recalculate_objective(solution["variables"])
                print(f"Solução Inteira Encontrada! Valor: {obj_val:.4f}")
                if obj_val > self.lower_bound:
                    print(f"*** Nova melhor solução encontrada! Lower bound atualizado para {obj_val:.4f} ***")
                    self.lower_bound = obj_val
                    solution['objective_value'] = obj_val
                    self.best_integer_solution = solution
                continue

            # Se, mesmo após os cortes, a solução ainda for fracionária, ramificamos.
            branch_var_name = self._find_branching_variable(solution["variables"], current_problem)
            if not branch_var_name:
                continue

            branch_var_value = solution["variables"][branch_var_name]
            print(f"Ramificando em '{branch_var_name}' com valor {branch_var_value:.4f}")
            
            problem_down = self._add_bound_to_problem(current_problem, branch_var_name, "<=", np.floor(branch_var_value))
            problem_up = self._add_bound_to_problem(current_problem, branch_var_name, ">=", np.floor(branch_var_value) + 1)
            
            self.node_queue.extend([problem_down, problem_up])

        # ... (O resto da função, com a impressão final, permanece o mesmo) ...

        print("\n--- Fim da Execução do Branch and Bound ---")
        if self.best_integer_solution:
             print("\n--- MELHOR SOLUÇÃO INTEIRA ENCONTRADA ---")
             solution_to_print = self.best_integer_solution
             print(f"Status: {solution_to_print['status']}")
             print(f"Valor da Função Objetivo: {solution_to_print['objective_value']:.4f}")
             print("Valores das Variáveis:")
             for var_name in self.root_problem.variable_names:
                 value = solution_to_print["variables"].get(var_name, 0.0)
                 if abs(value) > TOLERANCE:
                     # Arredondar o valor da variável para a impressão
                     print(f"  {var_name} = {round(value):.4f}")
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

    def _add_constraint_to_problem(self, problem: Problem, coeffs: dict, sense_str: str, value: float) -> Problem:
        new_problem = copy.deepcopy(problem)
        new_problem.name = f"{problem.name}_cut"
        num_vars = len(new_problem.variable_names)

        new_row = np.zeros((1, num_vars))
        for var_name, coeff in coeffs.items():
            var_idx = new_problem.variable_names.index(var_name)
            new_row[0, var_idx] = coeff
        
        new_problem.constraint_matrix = sp.vstack([new_problem.constraint_matrix.tocsr(), new_row]).tocsr()
        new_problem.rhs_vector = np.append(new_problem.rhs_vector, value)
        new_problem.constraint_senses.append(ConstraintSense(sense_str))
        return new_problem

    def _is_var_binary(self, var_name):
        var_idx = self.root_problem.variable_names.index(var_name)
        return (self.root_problem.lower_bounds[var_idx] == 0 and self.root_problem.upper_bounds[var_idx] == 1)

    def _is_integer_feasible(self, solution_vars: dict, problem: Problem) -> bool:
        for var_idx in problem.integer_variables:
            var_name = problem.variable_names[var_idx]
            value = solution_vars.get(var_name, 0.0)
            
            # Verifica se está suficientemente perto de um inteiro
            if abs(value - round(value)) > TOLERANCE: return False
            
            # Para binárias, verificamos se está perto de 0 ou 1
            if self._is_var_binary(var_name):
                if not (abs(value - 0) < TOLERANCE or abs(value - 1) < TOLERANCE):
                    return False
        return True
    
    def _find_branching_variable(self, solution_vars: dict, problem: Problem) -> str:
        for var_idx in problem.integer_variables:
            var_name = problem.variable_names[var_idx]
            value = solution_vars.get(var_name, 0.0)

            if abs(value - round(value)) > TOLERANCE:
                return var_name

            if self._is_var_binary(var_name):
                if not (abs(value - 0) < TOLERANCE or abs(value - 1) < TOLERANCE):
                    return var_name
        return None 
    
    def _recalculate_objective(self, solution_vars):
        obj_val = 0
        for var, val in solution_vars.items():
            if abs(val) > 1e-6:
                idx = self.root_problem.variable_names.index(var)
                obj_val += self.root_problem.objective_coeffs[idx] * val
        return obj_val