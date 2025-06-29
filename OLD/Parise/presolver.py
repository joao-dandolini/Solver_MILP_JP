# presolver.py
import math
import copy
from collections import defaultdict # NOVO: Importa defaultdict para as estatísticas
from mip_problem import MIPProblem, Constraint

class InfeasibleProblemError(Exception):
    pass

class Presolver:
    def __init__(self, problem: MIPProblem, use_probing: bool = True, probe_limit: int = 50):
        self.problem = problem.copy()
        self.modifications = 0
        self.vars_map = {var.name: var for var in self.problem.variables}
        self.tolerance = 1e-9
        
        self.use_probing = use_probing
        self.probe_limit = probe_limit
        
        # --- NOVO: Dicionário para armazenar estatísticas ---
        self.stats = defaultdict(int)

    def _print_summary(self):
        """Imprime um resumo das modificações feitas pelo presolve."""
        print("\n--- Resumo do Presolve ---")
        if not self.stats:
            print("Nenhuma modificação foi realizada.")
            print("--------------------------")
            return
        
        total_modifications = sum(self.stats.values())
        print(f"Total de modificações: {total_modifications}")
        print("---------------------------------------------")
        print(f"{'Técnica':<25s} | {'Modificações':>15s}")
        print("-------------------------|------------------")
        for technique, count in sorted(self.stats.items()):
            print(f"{technique:<25s} | {count:>16d}")
        print("---------------------------------------------")

    def presolve(self) -> MIPProblem:
        print("--- Iniciando a rotina de Presolve ---")
        round_num = 1
        while True:
            self.modifications = 0
            
            self._apply_singleton_updates()
            self._apply_bound_propagation()
            self._apply_redundancy_removal()
            
            if self.use_probing:
                self._probe()

            try:
                self._check_infeasibility(self.vars_map)
            except InfeasibleProblemError:
                print("\n!!! Presolve detectou que o problema é INVIÁVEL !!!")
                self._print_summary() # Imprime o resumo mesmo em caso de erro
                raise

            print(f"Rodada {round_num} de Presolve completada. Modificações: {self.modifications}")
            if self.modifications == 0:
                break
            round_num += 1
            
        print("--- Presolve finalizado ---")
        self._print_summary() # Imprime o resumo final
        return self.problem

    def _apply_singleton_updates(self):
        removable_constraints, bound_updates = self._get_singleton_updates(
            self.problem.constraints, self.vars_map
        )
        
        if bound_updates:
            num_bounds_changed = 0
            for var_name, bound_type, value in bound_updates:
                if bound_type == 'lb':
                    if value > self.vars_map[var_name].lb + self.tolerance:
                         self.vars_map[var_name].lb = value
                         num_bounds_changed += 1
                else:
                    if value < self.vars_map[var_name].ub - self.tolerance:
                        self.vars_map[var_name].ub = value
                        num_bounds_changed += 1
            if num_bounds_changed > 0:
                self.stats['Bounds por Singletons'] += num_bounds_changed
                self.modifications += num_bounds_changed
        
        if removable_constraints:
            num_removed = len(removable_constraints)
            print(f"  - Presolve/Singletons: Removendo {num_removed} restrições.")
            self.problem.constraints = [c for c in self.problem.constraints if c not in removable_constraints]
            self.stats['Restrições (Singleton)'] += num_removed
            self.modifications += num_removed

    def _apply_bound_propagation(self):
        mods = self._propagate_bounds(self.vars_map, self.problem.constraints)
        if mods > 0:
            print(f"  - Presolve/Propagation: Apertou {mods} bounds de variáveis.")
            self.stats['Bounds por Propagação'] += mods
            self.modifications += mods

    def _apply_redundancy_removal(self):
        active_constraints = self._get_non_redundant_constraints(self.problem.constraints, self.vars_map)
        num_removed = len(self.problem.constraints) - len(active_constraints)
        if num_removed > 0:
            self.stats['Restrições Redundantes'] += num_removed
            self.modifications += num_removed
            self.problem.constraints = active_constraints

    def _probe(self):
        all_binary_vars = [var for var in self.problem.variables if var.is_integer and var.lb == 0 and var.ub == 1 and (var.ub - var.lb) > self.tolerance]
        binary_vars_to_probe = all_binary_vars[:self.probe_limit]
        if not binary_vars_to_probe: return
        print(f"  - Presolve/Probing: Analisando {len(binary_vars_to_probe)} de {len(all_binary_vars)} variáveis binárias candidatas.")

        for var in binary_vars_to_probe:
            probe_down_map = copy.deepcopy(self.vars_map)
            probe_down_map[var.name].ub = 0.0
            try:
                self._run_probe_propagation_loop(probe_down_map, self.problem.constraints)
                self._check_infeasibility(probe_down_map)
            except InfeasibleProblemError:
                if self.vars_map[var.name].lb < 1.0:
                    print(f"  -> Probing fixou '{var.name}' = 1")
                    self.vars_map[var.name].lb = 1.0
                    self.modifications += 1
                    self.stats['Fixações por Probing'] += 1
                continue

            probe_up_map = copy.deepcopy(self.vars_map)
            probe_up_map[var.name].lb = 1.0
            try:
                self._run_probe_propagation_loop(probe_up_map, self.problem.constraints)
                self._check_infeasibility(probe_up_map)
            except InfeasibleProblemError:
                if self.vars_map[var.name].ub > 0.0:
                    print(f"  -> Probing fixou '{var.name}' = 0")
                    self.vars_map[var.name].ub = 0.0
                    self.modifications += 1
                    self.stats['Fixações por Probing'] += 1

    # O resto dos métodos (_run_probe_propagation_loop, _get_singleton_updates, etc.)
    # não precisam de alteração, pois a contagem é feita nos métodos `_apply_...` e `_probe`.
    # O código completo deles está incluído abaixo para garantir a integridade.
    def _run_probe_propagation_loop(self, probe_vars_map, constraints):
        for _ in range(5): 
            if self._propagate_bounds(probe_vars_map, constraints) == 0: break

    def _get_singleton_updates(self, constraints, vars_map):
        removable_constraints, bound_updates = [], []
        for const in constraints:
            if len(const.coeffs) == 1:
                var_name, coeff = list(const.coeffs.items())[0]
                if abs(coeff) < self.tolerance: removable_constraints.append(const); continue
                nbv = const.rhs / coeff
                if const.sense == '<=':
                    if coeff > 0: bound_updates.append((var_name, 'ub', nbv))
                    else: bound_updates.append((var_name, 'lb', nbv))
                elif const.sense == '>=':
                    if coeff > 0: bound_updates.append((var_name, 'lb', nbv))
                    else: bound_updates.append((var_name, 'ub', nbv))
                elif const.sense == '==':
                    bound_updates.append((var_name, 'lb', nbv)); bound_updates.append((var_name, 'ub', nbv))
                removable_constraints.append(const)
        return removable_constraints, bound_updates

    def _get_non_redundant_constraints(self, constraints, vars_map):
        active_constraints, num_removed = [], 0
        for const in constraints:
            min_activity, max_activity = self._calculate_activity_bounds(const, vars_map)
            if abs(min_activity) == float('inf') or abs(max_activity) == float('inf'):
                active_constraints.append(const); continue
            is_redundant = False
            if const.sense == '<=' and max_activity <= const.rhs + self.tolerance: is_redundant = True
            elif const.sense == '>=' and min_activity >= const.rhs - self.tolerance: is_redundant = True
            if is_redundant: num_removed += 1
            else: active_constraints.append(const)
        if num_removed > 0: print(f"  - Presolve/Redundancy: Identificadas {num_removed} restrições redundantes para remoção.")
        return active_constraints

    def _propagate_bounds(self, vars_map, constraints) -> int:
        modifications_in_run = 0
        for const in constraints:
            for var_name_to_tighten, target_coeff in const.coeffs.items():
                if abs(target_coeff) < self.tolerance: continue
                target_var = vars_map[var_name_to_tighten]
                min_activity_others, max_activity_others = self._calculate_activity_bounds(const, vars_map, exclude_var=var_name_to_tighten)
                if abs(min_activity_others) == float('inf') or abs(max_activity_others) == float('inf'): continue
                if const.sense == '<=':
                    residual_rhs = const.rhs - min_activity_others
                    if target_coeff > 0:
                        if (nb := residual_rhs / target_coeff) < target_var.ub - self.tolerance: target_var.ub = nb; modifications_in_run += 1
                    else:
                        if (nb := residual_rhs / target_coeff) > target_var.lb + self.tolerance: target_var.lb = nb; modifications_in_run += 1
                elif const.sense == '>=':
                    residual_rhs = const.rhs - max_activity_others
                    if target_coeff > 0:
                        if (nb := residual_rhs / target_coeff) > target_var.lb + self.tolerance: target_var.lb = nb; modifications_in_run += 1
                    else:
                        if (nb := residual_rhs / target_coeff) < target_var.ub - self.tolerance: target_var.ub = nb; modifications_in_run += 1
        return modifications_in_run

    def _calculate_activity_bounds(self, constraint, vars_map, exclude_var=None):
        min_activity, max_activity = 0, 0
        for var_name, coeff in constraint.coeffs.items():
            if var_name == exclude_var: continue
            var = vars_map[var_name]
            if coeff > 0: min_activity += coeff * var.lb; max_activity += coeff * var.ub
            else: min_activity += coeff * var.ub; max_activity += coeff * var.lb
        return min_activity, max_activity

    def _check_infeasibility(self, vars_map):
        for var in vars_map.values():
            if var.lb > var.ub + self.tolerance: raise InfeasibleProblemError(f"Inviabilidade na variável '{var.name}': lb ({var.lb}) > ub ({var.ub})")