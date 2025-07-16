# presolve.py (Versão Final, baseada no código de referência funcional)

import gurobipy as gp
from gurobipy import GRB
import math
from typing import Dict, List

# --- TÉCNICA 1: FIXAR VARIÁVEIS DE RESTRIÇÕES "SINGLETON" ---
def _fix_from_singletons(model: gp.Model) -> int:
    """Aperta os bounds usando restrições com apenas uma variável."""
    changes = 0
    constrs_to_remove = []
    for constr in model.getConstrs():
        if model.getRow(constr).size() != 1: continue
        
        row = model.getRow(constr)
        var, coeff, rhs, sense = row.getVar(0), row.getCoeff(0), constr.RHS, constr.Sense
        
        if abs(coeff) < 1e-9: continue
        constrs_to_remove.append(constr)
        implied_val = rhs / coeff
        
        try:
            if sense == GRB.LESS_EQUAL:
                if coeff > 0:
                    if implied_val < var.UB: var.UB = implied_val; changes += 1
                else:
                    if implied_val > var.LB: var.LB = implied_val; changes += 1
            elif sense == GRB.GREATER_EQUAL:
                if coeff > 0:
                    if implied_val > var.LB: var.LB = implied_val; changes += 1
                else:
                    if implied_val < var.UB: var.UB = implied_val; changes += 1
            elif sense == GRB.EQUAL:
                if var.LB != implied_val or var.UB != implied_val:
                    var.LB = implied_val; var.UB = implied_val; changes += 1
        except gp.GurobiError:
            return -1 # Inviabilidade
            
    for constr in constrs_to_remove: model.remove(constr)
    return changes

# --- TÉCNICA 2: PROPAGAÇÃO DE BOUNDS ---
def _propagate_bounds(model: gp.Model, original_vars: Dict[str, str]) -> int:
    """Itera sobre as restrições para apertar os limites das variáveis."""
    tightenings = 0
    TOLERANCE = 1e-7
    for constr in tuple(model.getConstrs()):
        row = model.getRow(constr)
        if row.size() < 2: continue
        rhs, sense = constr.RHS, constr.Sense
        for i in range(row.size()):
            var_i = row.getVar(i)
            if var_i.LB > var_i.UB - TOLERANCE: continue
            coeff_i = row.getCoeff(i)
            if abs(coeff_i) < TOLERANCE: continue
            
            activity_rest_min, activity_rest_max = 0.0, 0.0
            for j in range(row.size()):
                if i == j: continue
                var_j, coeff_j = row.getVar(j), row.getCoeff(j)
                if (coeff_j > 0 and var_j.LB == -GRB.INFINITY) or (coeff_j < 0 and var_j.UB == GRB.INFINITY): activity_rest_min = -GRB.INFINITY
                if (coeff_j > 0 and var_j.UB == GRB.INFINITY) or (coeff_j < 0 and var_j.LB == -GRB.INFINITY): activity_rest_max = GRB.INFINITY
                if activity_rest_min != -GRB.INFINITY: activity_rest_min += coeff_j * var_j.LB if coeff_j > 0 else coeff_j * var_j.UB
                if activity_rest_max != GRB.INFINITY: activity_rest_max += coeff_j * var_j.UB if coeff_j > 0 else coeff_j * var_j.LB
            
            new_ub = var_i.UB
            if sense in [GRB.LESS_EQUAL, GRB.EQUAL] and coeff_i > 0 and activity_rest_min > -GRB.INFINITY: new_ub = min(new_ub, (rhs - activity_rest_min) / coeff_i)
            elif sense in [GRB.GREATER_EQUAL, GRB.EQUAL] and coeff_i < 0 and activity_rest_max < GRB.INFINITY: new_ub = min(new_ub, (rhs - activity_rest_max) / coeff_i)
            if new_ub < var_i.UB - TOLERANCE:
                final_ub = math.floor(new_ub + TOLERANCE) if original_vars.get(var_i.VarName) != GRB.CONTINUOUS else new_ub
                if final_ub < var_i.LB - TOLERANCE: return -1
                var_i.UB = final_ub
                tightenings += 1

            new_lb = var_i.LB
            if sense in [GRB.GREATER_EQUAL, GRB.EQUAL] and coeff_i > 0 and activity_rest_max < GRB.INFINITY: new_lb = max(new_lb, (rhs - activity_rest_max) / coeff_i)
            elif sense in [GRB.LESS_EQUAL, GRB.EQUAL] and coeff_i < 0 and activity_rest_min > -GRB.INFINITY: new_lb = max(new_lb, (rhs - activity_rest_min) / coeff_i)
            if new_lb > var_i.LB + TOLERANCE:
                final_lb = math.ceil(new_lb - TOLERANCE) if original_vars.get(var_i.VarName) != GRB.CONTINUOUS else new_lb
                if final_lb > var_i.UB + TOLERANCE: return -1
                var_i.LB = final_lb
                tightenings += 1
            
            if var_i.LB > var_i.UB + TOLERANCE: return -1
    return tightenings

# --- TÉCNICA 3: PROBING SEGURO ---
def _probe_binary_variables(model: gp.Model, original_vars: Dict[str, str]) -> int:
    """Executa probing em variáveis binárias, coletando e aplicando as mudanças no final."""
    changes = 0
    binary_vars = [v for v in model.getVars() if original_vars.get(v.VarName) == GRB.BINARY and v.LB != v.UB]
    if not binary_vars: return 0

    print(f"INFO: [Presolve-Probing] Sondando {len(binary_vars)} variáveis binárias...")
    
    vars_to_fix_to_0, vars_to_fix_to_1 = [], []
    probe_model = None
    try:
        probe_model = model.copy()
        probe_model.setParam('OutputFlag', 0)
        probe_model.setParam('TimeLimit', 1)

        for var in binary_vars:
            p_var = probe_model.getVarByName(var.VarName)
            
            original_lb, original_ub = p_var.LB, p_var.UB
            
            p_var.LB = 1.0
            probe_model.optimize()
            if probe_model.Status == GRB.INFEASIBLE: vars_to_fix_to_0.append(var.VarName)
            p_var.LB = original_lb

            if var.VarName not in vars_to_fix_to_0:
                p_var.UB = 0.0
                probe_model.optimize()
                if probe_model.Status == GRB.INFEASIBLE: vars_to_fix_to_1.append(var.VarName)
                p_var.UB = original_ub
    finally:
        if probe_model: probe_model.dispose()

    for var_name in vars_to_fix_to_0:
        var_obj = model.getVarByName(var_name)
        if var_obj.UB != 0.0: var_obj.UB = 0.0; changes += 1
    for var_name in vars_to_fix_to_1:
        var_obj = model.getVarByName(var_name)
        if var_obj.LB != 1.0: var_obj.LB = 1.0; changes += 1
    return changes

# --- ORQUESTRADOR PRINCIPAL ---

def run_presolve(model: gp.Model, original_vars: Dict[str, str]) -> str:
    """Orquestra as rotinas de presolve, baseado na estrutura do código de referência."""
    print("-" * 60)
    print("INFO: [Presolve] Iniciando fase de pré-processamento...")
    
    for i in range(1): # No código de referência, ele roda apenas uma vez
        print(f"INFO: [Presolve] Passada {i+1}...")
        
        # Ordem lógica inspirada no código de referência
        changes = _fix_from_singletons(model)
        if changes == -1: return 'INFEASIBLE'
        
        changes = _propagate_bounds(model, original_vars)
        if changes == -1: return 'INFEASIBLE'

        # Probing é executado no final
        _probe_binary_variables(model, original_vars)
        model.update()
        
    print("INFO: [Presolve] Fase de pré-processamento concluída.")
    return 'SUCCESS'