# heuristics.py
# Implementação final e PURA da Feasibility Pump 2.0.

from typing import Dict, Any, Optional, List, Tuple
import gurobipy as gp
from gurobipy import GRB, quicksum
import math
from collections import deque
import random

# --- NOVA FUNÇÃO AUXILIAR DE VERIFICAÇÃO ---
def _is_solution_feasible(
    model: gp.Model, 
    solution: Dict[str, float]
) -> bool:
    """Verifica manualmente se uma dada solução inteira satisfaz todas as restrições."""
    TOLERANCE = 1e-6
    for constr in model.getConstrs():
        lhs_expr = model.getRow(constr)
        lhs_value = 0.0
        for i in range(lhs_expr.size()):
            var = lhs_expr.getVar(i)
            coeff = lhs_expr.getCoeff(i)
            lhs_value += coeff * solution.get(var.VarName, 0.0)

        if constr.Sense == GRB.LESS_EQUAL and lhs_value > constr.RHS + TOLERANCE:
            return False
        elif constr.Sense == GRB.GREATER_EQUAL and lhs_value < constr.RHS - TOLERANCE:
            return False
        elif constr.Sense == GRB.EQUAL and abs(lhs_value - constr.RHS) > TOLERANCE:
            return False
            
    return True

# As 3 funções de propagação (inalteradas)
def _propagate_knapsack_constraint(knapsack_vars, rhs, domains):
    min_activity = sum(item['coeff'] * domains[item['var'].VarName]['lb'] for item in knapsack_vars)
    newly_tightened_bounds = {}
    for item in knapsack_vars:
        var_name = item['var'].VarName
        if domains[var_name]['ub'] == 0.0: continue
        coeff = item['coeff']
        potential_activity = min_activity - (coeff * domains[var_name]['lb']) + (coeff * 1.0)
        if potential_activity > rhs + 1e-6:
            newly_tightened_bounds[var_name] = {'ub': 0.0}
    return newly_tightened_bounds

def _run_propagation_engine(knapsack_constraints, var_to_constrs_map, domains):
    propagation_queue = deque(knapsack_constraints.keys())
    tightened_domains = {k: v.copy() for k, v in domains.items()}
    while propagation_queue:
        constr_idx = propagation_queue.popleft()
        if constr_idx not in knapsack_constraints: continue
        constr_data = knapsack_constraints[constr_idx]
        new_bounds = _propagate_knapsack_constraint(constr_data['vars'], constr_data['rhs'], tightened_domains)
        if new_bounds:
            for var_name, bound_info in new_bounds.items():
                if tightened_domains[var_name]['ub'] != 0.0:
                    tightened_domains[var_name].update(bound_info)
                    for neighbor_constr_idx in var_to_constrs_map.get(var_name, []):
                        if neighbor_constr_idx not in propagation_queue:
                            propagation_queue.append(neighbor_constr_idx)
    return tightened_domains

def _propagation_based_rounding(model, original_vars, lp_solution, knapsack_constraints, var_to_constrs_map):
    domains = {v.VarName: {'lb': v.LB, 'ub': v.UB} for v in model.getVars()}
    integer_vars = {v_name for v_name, v_type in original_vars.items() if v_type != GRB.CONTINUOUS}
    unfixed_vars = list(integer_vars)
    while unfixed_vars:
        unfixed_vars.sort(key=lambda v: 0.5 - abs(lp_solution.get(v, 0.5) - math.floor(lp_solution.get(v, 0.5)) - 0.5))
        var_to_fix = unfixed_vars.pop(0)
        rounded_value = round(lp_solution.get(var_to_fix, 0.0))
        domains[var_to_fix]['lb'] = rounded_value
        domains[var_to_fix]['ub'] = rounded_value
        domains = _run_propagation_engine(knapsack_constraints, var_to_constrs_map, domains)
        unfixed_vars = [v for v in integer_vars if domains[v]['lb'] != domains[v]['ub']]
    return {v_name: val['lb'] for v_name, val in domains.items()}

# Em heuristics.py

def _solve_lookahead_lp(model: gp.Model, var_to_fix: str, fix_value: float) -> Optional[float]:
    """Função auxiliar para testar um único branch durante o mergulho."""
    temp_model = model.copy()
    temp_model.setParam(GRB.Param.OutputFlag, 0)
    
    var_obj = temp_model.getVarByName(var_to_fix)
    var_obj.lb = fix_value
    var_obj.ub = fix_value
    
    temp_model.optimize()
    
    if temp_model.Status == GRB.OPTIMAL:
        return temp_model.ObjVal
    return None

def run_diving_heuristic(
    model: gp.Model, 
    original_vars: Dict[str, str],
    max_dive_depth: int = 100 # Reduzimos a profundidade padrão para ser mais rápido
) -> Optional[Dict[str, Any]]:
    """
    Executa uma heurística de "Mergulho com Lookahead".
    A cada passo, testa os ramos de 'subir' e 'descer' e escolhe o que
    degrada menos a função objetivo.
    """
    print("-" * 60)
    print("INFO: Iniciando Heurística de Mergulho com Lookahead...")

    work_model = model.copy()
    work_model.setParam(GRB.Param.Presolve, 0)
    work_model.setParam(GRB.Param.Cuts, 0)
    work_model.setParam('OutputFlag', 0)
    
    # Relaxa o modelo de trabalho para a heurística
    for v in work_model.getVars():
        if original_vars.get(v.VarName) != GRB.CONTINUOUS:
            v.VType = GRB.CONTINUOUS
    work_model.update()
    
    original_objective = model.getObjective() # Pega o objetivo do modelo original
    
    # Loop principal do mergulho
    for depth in range(max_dive_depth):
        work_model.optimize()

        if work_model.Status != GRB.OPTIMAL:
            print("INFO: [Lookahead Dive] LP tornou-se inviável. Parando.")
            return None

        lp_solution = {v.VarName: v.X for v in work_model.getVars()}
        
        fractional_vars = {
            name: val for name, val in lp_solution.items()
            if original_vars.get(name) != GRB.CONTINUOUS and abs(val - round(val)) > 1e-6
        }

        if not fractional_vars:
            print("INFO: [Lookahead Dive] SUCESSO! Solução inteira encontrada.")
            final_solution = {v.VarName: v.X for v in work_model.getVars()}
            # Pega o valor do objetivo diretamente do modelo que foi resolvido
            obj_val = work_model.ObjVal # <-- CORREÇÃO
            return {'solution': final_solution, 'objective': obj_val}

        # Seleciona a variável mais fracionária para analisar
        var_to_fix_name = max(
            fractional_vars.keys(), 
            key=lambda k: 0.5 - abs(fractional_vars[k] - math.floor(fractional_vars[k]) - 0.5)
        )
        
        var_value = fractional_vars[var_to_fix_name]
        
        # --- LÓGICA DO LOOKAHEAD ---
        # Testa os dois caminhos: arredondar para baixo e para cima
        obj_down = _solve_lookahead_lp(work_model, var_to_fix_name, math.floor(var_value))
        obj_up = _solve_lookahead_lp(work_model, var_to_fix_name, math.ceil(var_value))
        
        # Escolhe o caminho que leva a uma solução melhor (ou menos pior)
        # Para um problema de MIN, queremos o menor ObjVal. Para MAX, o maior.
        is_minimization = model.ModelSense == GRB.MINIMIZE
        
        # Decide qual ramo é mais promissor
        if obj_down is None and obj_up is None:
            print(f"INFO: [Lookahead Dive] Ambos os ramos para '{var_to_fix_name}' são inviáveis. Parando.")
            return None
        
        go_down = False
        if obj_down is not None and obj_up is not None:
            go_down = obj_down < obj_up if is_minimization else obj_down > obj_up
        elif obj_down is not None:
            go_down = True
            
        fix_value = math.floor(var_value) if go_down else math.ceil(var_value)
        direction = "'baixo'" if go_down else "'cima'"
        # --- FIM DA LÓGICA DO LOOKAHEAD ---

        print(f"INFO: [Lookahead Dive] Profundidade {depth+1}: Fixando '{var_to_fix_name}' para {fix_value} (ramo mais promissor: {direction})")

        # Fixa a variável no modelo de trabalho para a próxima iteração
        var_obj = work_model.getVarByName(var_to_fix_name)
        var_obj.lb = fix_value
        var_obj.ub = fix_value

    print("INFO: [Lookahead Dive] Profundidade máxima atingida.")
    return None

def run_feasibility_pump(
    model: gp.Model, 
    original_vars: Dict[str, str],
    max_iterations: int = 30,
    stagnation_limit: int = 5, # Aumentando um pouco o limite
    num_vars_to_flip: int = 20
) -> Optional[Dict[str, Any]]:
    """
    Executa a Feasibility Pump 2.0 com reinicialização guiada e log completo.
    """
    print("-" * 60)
    print("INFO: Iniciando Heurística Feasibility Pump 2.0...")

    # ... (código de setup inicial inalterado) ...
    original_objective = model.getObjective()
    work_model = model.copy()
    work_model.setParam('OutputFlag', 0)
    work_model.setParam(GRB.Param.Presolve, 0)
    work_model.setParam(GRB.Param.Cuts, 0)
    work_model.update()
    
    var_to_constrs_map = {v_name: [] for v_name in original_vars}
    knapsack_constraints = {}
    for i, constr in enumerate(work_model.getConstrs()):
        if constr.Sense not in [GRB.LESS_EQUAL, GRB.EQUAL]: continue
        lhs = work_model.getRow(constr)
        is_candidate = all(original_vars.get(lhs.getVar(j).VarName) == GRB.BINARY and lhs.getCoeff(j) >= 1e-6 for j in range(lhs.size()))
        if is_candidate:
            knapsack_constraints[i] = {'vars': [{'var': lhs.getVar(j), 'coeff': lhs.getCoeff(j)} for j in range(lhs.size())], 'rhs': constr.RHS}
            for j in range(lhs.size()):
                var_to_constrs_map[lhs.getVar(j).VarName].append(i)

    work_model.optimize()
    if work_model.Status != GRB.OPTIMAL: return None
    
    x_lp = {v.VarName: v.X for v in work_model.getVars()}
    integer_vars = [work_model.getVarByName(v) for v, type in original_vars.items() if type != GRB.CONTINUOUS]
    TOLERANCE = 1e-6
    last_distance = float('inf')
    stagnation_counter = 0

    for i in range(max_iterations):
        print(f"\nINFO: [FP2.0] Iteração Principal {i+1}")
        
        x_i = _propagation_based_rounding(work_model, original_vars, x_lp, knapsack_constraints, var_to_constrs_map)
        
        # --- CÁLCULO E PRINT DA DISTÂNCIA RESTAURADOS ---
        distance = sum(abs(x_lp.get(v.VarName, 0.0) - x_i.get(v.VarName, 0.0)) for v in integer_vars)
        print(f"INFO: [FP2.0] Distância da iteração: {distance:.4f}")
        # --- FIM DA RESTAURAÇÃO ---

        if _is_solution_feasible(model, x_i):
            print("INFO: [FP2.0] SUCESSO! Solução inteira encontrada e verificada como viável.")
            obj_val = original_objective.getValue(x_i)
            return {'solution': x_i, 'objective': obj_val}

        if abs(distance - last_distance) < TOLERANCE:
            stagnation_counter += 1
        else: 
            stagnation_counter = 0
            last_distance = distance

        if stagnation_counter >= stagnation_limit:
            print(f"INFO: [FP2.0] Distância estagnada. Aplicando perturbação guiada...")
            conflicts = []
            for var in integer_vars:
                dist = abs(x_lp.get(var.VarName, 0.0) - x_i.get(var.VarName, 0.0))
                conflicts.append({'name': var.VarName, 'dist': dist})
            conflicts.sort(key=lambda x: x['dist'], reverse=True)
            
            x_lp_perturbed = x_lp.copy()
            for k in range(min(num_vars_to_flip, len(conflicts))):
                var_to_flip = conflicts[k]['name']
                x_lp_perturbed[var_to_flip] = 1.0 - x_lp_perturbed[var_to_flip]
            
            x_lp = x_lp_perturbed
            stagnation_counter = 0
            print(f"INFO: [FP2.0] Solução LP perturbada para guiar a próxima iteração.")
            continue # Pula para a próxima iteração com a nova x_lp
            
        distance_objective = quicksum(v if x_i.get(v.VarName, 0) == 0 else (1 - v) for v in integer_vars if original_vars.get(v.VarName) == GRB.BINARY)
        work_model.setObjective(distance_objective, GRB.MINIMIZE)
        work_model.optimize()

        if work_model.Status != GRB.OPTIMAL:
            print("INFO: [FP2.0] O LP da bomba tornou-se inviável."); break
        
        x_lp = {v.VarName: v.X for v in work_model.getVars()}

    print("INFO: [FP2.0] Heurística concluída sem encontrar solução viável.")
    return None

def run_rins(
    original_model: gp.Model,
    incumbent_solution: Dict[str, float],
    current_lp_solution: Dict[str, float],
    incumbent_objective: float, # Novo parâmetro
    time_limit_seconds: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Executa a heurística RINS para tentar melhorar uma solução incumbente existente.

    Args:
        original_model: O modelo MILP original para criar o subproblema.
        incumbent_solution: A melhor solução inteira encontrada até agora.
        current_lp_solution: A solução LP do nó atual na árvore de B&B.
        time_limit_seconds: Limite de tempo para resolver o sub-MIP.

    Returns:
        Um dicionário com uma solução melhorada, se encontrada.
    """
    #print("INFO: [RINS] Tentando refinar a solução incumbente...")

    sub_mip = original_model.copy()
    sub_mip.setParam('OutputFlag', 0)
    sub_mip.setParam(GRB.Param.TimeLimit, time_limit_seconds)
    
    # Identifica as variáveis de consenso e as fixa
    fixed_vars_count = 0
    TOLERANCE = 1e-6
    for var_name, incumbent_val in incumbent_solution.items():
        lp_val = current_lp_solution.get(var_name)
        if lp_val is not None and abs(incumbent_val - lp_val) < TOLERANCE:
            var = sub_mip.getVarByName(var_name)
            var.lb = incumbent_val
            var.ub = incumbent_val
            fixed_vars_count += 1
            
    #print(f"INFO: [RINS] {fixed_vars_count} variáveis fixadas. Resolvendo sub-MIP...")
    
    sub_mip.optimize()

    # Se uma nova solução melhor for encontrada, a retorna
    if sub_mip.SolCount > 0 and sub_mip.ObjVal < incumbent_objective:
        #print(f"INFO: [RINS] SUCESSO! Solução melhorada encontrada com objetivo: {sub_mip.ObjVal:.4f}")
        new_solution = {v.VarName: v.X for v in sub_mip.getVars()}
        return {'solution': new_solution, 'objective': sub_mip.ObjVal}
        
    #print("INFO: [RINS] Nenhuma solução melhor encontrada no tempo limite.")
    return None