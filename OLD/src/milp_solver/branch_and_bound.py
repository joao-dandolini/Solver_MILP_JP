import copy
import numpy as np
import scipy.sparse as sp
import heapq
from src.core.problem import Problem, ConstraintSense
from src.lp_solver.simplex import SimplexSolver
from.cuts import generate_gomory_cut
from .heuristics import rounding_heuristic
from .presolve_adapter import convert_problem_to_presolver_format, convert_presolver_to_problem_format
from .clique_manager import initialize_conflict_graph, separate_clique_cuts
from src.presolve.MIP_presolver import MIPPresolver

TOLERANCE = 1e-6

class MILPSolver:
    def __init__(self, problem: Problem):
        self.root_problem = problem
        self.node_queue = []  # Agora é uma fila de prioridade (heap)
        self.best_integer_solution = None
        self.lower_bound = -float('inf')
        self.upper_bound = float('inf')
        self.node_count = 0  # Usado para desempate na fila
        self.conflict_graph = None

    def solve(self):
        print("--- Iniciando o Solver Branch and Bound ---")
        # =================================================================
        # ETAPA 1: PRÉ-PROCESSAMENTO (PRESOLVE)
        # =================================================================
        print("\n--- Fase 1: Pré-processamento (Presolve) ---")
        try:
            presolve_vars, presolve_constrs = convert_problem_to_presolver_format(self.root_problem)
            presolver = MIPPresolver(presolve_vars, presolve_constrs)

            # --- BLOCO DE DIAGNÓSTICO UNIVERSAL: ANTES ---
            print("\n  Estado do Problema ANTES do Presolve:")
            print(f"    - Número de Variáveis: {len(presolver.variables)}")
            print(f"    - Número de Restrições: {len(presolver.constraints)}")
            print("    - Limites das primeiras variáveis:")
            vars_to_print_before = list(presolver.variables.keys())
            for var_name in vars_to_print_before:
                info = presolver.variables[var_name]
                print(f"      - {info['lb']} <= {var_name} <= {info['ub']}")
            # ---------------------------------------------

            print("\n  Executando métodos de Presolve...")
            presolver.bound_propagation()
            # Adicione outras chamadas aqui se desejar testá-las
            print("  ...Presolve concluído.")

            # --- BLOCO DE DIAGNÓSTICO UNIVERSAL: DEPOIS ---
            print("\n  Estado do Problema DEPOIS do Presolve:")
            print(f"    - Número de Variáveis: {len(presolver.variables)}")
            print(f"    - Número de Restrições: {len(presolver.constraints)}")
            print("    - Limites das primeiras variáveis (após presolve):")
            vars_to_print_after = list(presolver.variables.keys())
            if not vars_to_print_after:
                print("      - Nenhuma variável restante após o presolve.")
            else:
                for var_name in vars_to_print_after:
                    info = presolver.variables[var_name]
                    print(f"      - {info['lb']} <= {var_name} <= {info['ub']}")
            # ----------------------------------------------
            
            # 1c. VOLTA: Traduzir de volta para o nosso formato Problem
            print("\n  Reconstruindo problema simplificado...")
            processed_problem = convert_presolver_to_problem_format(
                presolver.variables, presolver.constraints, self.root_problem
            )
            print("  Presolve concluído e problema reconstruído!")

        except Exception as e:
            print(f"  --> Erro durante o presolve: {e}. Continuando com o problema original.")
            processed_problem = self.root_problem

        # --- ETAPA 2: ANÁLISE ESTRUTURAL (CLIQUE) ---
        print("\n--- Fase 2: Análise Estrutural para Cortes de Clique ---")
        self.conflict_graph = initialize_conflict_graph(processed_problem)
        # =================================================================
        # ETAPA 3: RESOLVER O NÓ RAIZ E EXECUTAR HEURÍSTICA
        # =================================================================
        print("\n--- Fase 3: Resolvendo Nó Raiz e Executando Heurística ---")
        root_solver = SimplexSolver(processed_problem)
        root_solution_dict, _, _ = root_solver.solve()

        if not root_solution_dict or root_solution_dict["status"] != "Optimal":
            print("Problema raiz infactível ou ilimitado. Encerrando.")
            return None
            
        self.upper_bound = root_solution_dict["objective_value"]
        print(f"Limite Superior Inicial (Upper Bound): {self.upper_bound:.4f}")

        heuristic_solution = rounding_heuristic(processed_problem, root_solution_dict)
        if heuristic_solution:
            self.best_integer_solution = heuristic_solution
            self.lower_bound = self._recalculate_objective(heuristic_solution["variables"])
            print(f"*** Heurística definiu Lower Bound inicial para: {self.lower_bound:.4f} ***")
            if abs(self.upper_bound) > TOLERANCE:
                gap = (self.upper_bound - self.lower_bound) / abs(self.upper_bound)
                print(f"  ---> GAP INICIAL: {gap:.2%}")

        # =================================================================
        # ETAPA 4: BRANCH AND CUT
        # =================================================================
        self.node_queue.append(processed_problem) # Começa a fila com o problema (pós-presolve)
        
        # Verifica se a solução do nó raiz já é a resposta final
        if self._is_integer_feasible(root_solution_dict["variables"], processed_problem):
            obj_val = self._recalculate_objective(root_solution_dict["variables"])
            if obj_val > self.lower_bound:
                self.lower_bound = obj_val
                self.best_integer_solution = root_solution_dict
            print("Solução do nó raiz é inteira e ótima. Busca não é necessária.")
            self.node_queue = [] # Esvazia a fila

        iteration = 0
        max_b_and_b_iterations = 2000

        while self.node_queue:
            if iteration >= max_b_and_b_iterations:
                print("Número máximo de iterações B&B atingido.")
                break
            iteration += 1

            current_problem = self.node_queue.pop(0)
            problem_for_branching = copy.deepcopy(current_problem)
            print(f"\n--- Iteração B&B {iteration}: Resolvendo {current_problem.name} ---")

            # Para cada nó, tentamos adicionar até 10 cortes para fortalecê-lo
            max_cuts_per_node = 3
            for cut_iteration in range(max_cuts_per_node):
                print(f"  Rodada de cortes {cut_iteration + 1}/{max_cuts_per_node}...")
                
                lp_solver = SimplexSolver(current_problem)
                solution, final_tableau, final_basis = lp_solver.solve()
                
                if not solution or solution["status"] != "Optimal" or self._is_integer_feasible(solution["variables"], current_problem):
                    break

                cuts_found_this_pass = False
                
                # Tenta gerar cortes de Gomory
                gomory_cut = generate_gomory_cut(final_tableau, final_basis, lp_solver.variable_names, current_problem)
                if gomory_cut:
                    print("  --> Corte de Gomory encontrado!")
                    current_problem = self._add_constraint_to_problem(
                        current_problem, gomory_cut["coeffs"], gomory_cut["sense"], gomory_cut["rhs"]
                    )
                    cuts_found_this_pass = True

                # Tenta gerar cortes de Clique
                if self.conflict_graph:
                    clique_cuts = separate_clique_cuts(self.conflict_graph, solution["variables"], current_problem)
                    if clique_cuts:
                        print(f"  --> {len(clique_cuts)} corte(s) de clique encontrado(s)!")
                        for cut in clique_cuts:
                            current_problem = self._add_constraint_to_problem(
                                current_problem, cut["coeffs"], cut["sense"], cut["rhs"]
                            )
                        cuts_found_this_pass = True
                
                if not cuts_found_this_pass:
                    print("  Nenhum corte adicional encontrado nesta rodada.")
                    break
                else:
                    print("  Problema fortalecido com cortes. Re-otimizando...")

            # --- ANÁLISE DO RESULTADO FINAL DO NÓ (APÓS TENTATIVAS DE CORTE) ---
            if not solution or solution["status"] != "Optimal":
                print(f"Nó podado. Status final do LP: {solution.get('status', 'Falhou')}")
                continue

            #if iteration == 1:
            #    self.upper_bound = solution["objective_value"]
            #    print(f"Limite Superior Inicial (Upper Bound): {self.upper_bound:.4f}")

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

                    # Apenas calculamos e imprimimos o GAP quando encontramos uma solução melhor.
                    if self.upper_bound != float('inf') and abs(self.upper_bound) > 1e-9:
                        gap = (self.upper_bound - self.lower_bound) / abs(self.upper_bound)
                        print(f"  ---> NOVO GAP DE OTIMALIDADE: {gap:.2%}")
                        
                continue

            # Se, mesmo após os cortes, a solução ainda for fracionária, ramificamos.
            branch_var_name = self._find_branching_variable(solution["variables"], current_problem)
            if not branch_var_name:
                continue

            branch_var_value = solution["variables"][branch_var_name]
            print(f"Ramificando em '{branch_var_name}' com valor {branch_var_value:.4f}")
            
            problem_down = self._add_bound_to_problem(problem_for_branching, branch_var_name, "<=", np.floor(branch_var_value))
            problem_up = self._add_bound_to_problem(problem_for_branching, branch_var_name, ">=", np.floor(branch_var_value) + 1)
            
            self.node_queue.extend([problem_down, problem_up])

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