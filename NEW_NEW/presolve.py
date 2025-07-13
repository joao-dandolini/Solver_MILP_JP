# presolve.py
# Contém as rotinas de pré-processamento para simplificar o modelo.

from typing import Dict, Any, List
import gurobipy as gp
from gurobipy import GRB
import math

def _strengthen_bounds(model: gp.Model, original_vars: Dict[str, str]) -> int:
    """
    Executa UMA rodada de Fortalecimento de Limites no modelo.
    Técnica descrita na Seção 3.1 do artigo 'Presolve Reductions in MIP'.
    """
    bounds_changed_this_iteration = 0
    TOLERANCE = 1e-7

    for constr in model.getConstrs():
        if constr.Sense not in [GRB.LESS_EQUAL, GRB.GREATER_EQUAL]:
            continue

        lhs_expr = model.getRow(constr)
        rhs = constr.RHS
        
        min_activity, max_activity = 0.0, 0.0
        for i in range(lhs_expr.size()):
            var = lhs_expr.getVar(i)
            coeff = lhs_expr.getCoeff(i)
            if coeff > 0:
                min_activity += coeff * var.LB
                max_activity += coeff * var.UB
            else:
                min_activity += coeff * var.UB
                max_activity += coeff * var.LB
        
        for i in range(lhs_expr.size()):
            var = lhs_expr.getVar(i)
            coeff = lhs_expr.getCoeff(i)
            
            if abs(coeff) < TOLERANCE: continue

            if constr.Sense == GRB.LESS_EQUAL:
                folga = rhs - (min_activity - coeff * (var.LB if coeff > 0 else var.UB))
                if coeff > 0 and (new_bound := folga / coeff) < var.UB - TOLERANCE:
                    var.UB = math.floor(new_bound) if original_vars[var.VarName] != GRB.CONTINUOUS else new_bound
                    bounds_changed_this_iteration += 1
                elif coeff < 0 and (new_bound := folga / coeff) > var.LB + TOLERANCE:
                    var.LB = math.ceil(new_bound) if original_vars[var.VarName] != GRB.CONTINUOUS else new_bound
                    bounds_changed_this_iteration += 1
            
            elif constr.Sense == GRB.GREATER_EQUAL:
                folga = (max_activity - coeff * (var.UB if coeff > 0 else var.LB)) - rhs
                if coeff > 0 and (new_bound := -folga / coeff) > var.LB + TOLERANCE:
                    var.LB = math.ceil(new_bound) if original_vars[var.VarName] != GRB.CONTINUOUS else new_bound
                    bounds_changed_this_iteration += 1
                elif coeff < 0 and (new_bound := -folga / coeff) < var.UB - TOLERANCE:
                    var.UB = math.floor(new_bound) if original_vars[var.VarName] != GRB.CONTINUOUS else new_bound
                    bounds_changed_this_iteration += 1
                    
    if bounds_changed_this_iteration > 0:
        print(f"INFO: [Presolve-BS] Rodada de fortalecimento de limites apertou {bounds_changed_this_iteration} limites.")

    return bounds_changed_this_iteration

def _remove_redundant_constraints(model: gp.Model) -> int:
    """Encontra e remove restrições redundantes do modelo."""
    constraints_to_remove = [
        c for c in model.getConstrs() if not c.ConstrName.startswith("cover_cut") and _is_constraint_redundant(model, c)
    ]
    if constraints_to_remove:
        print(f"INFO: [Presolve-Redundant] Removendo {len(constraints_to_remove)} restrições redundantes.")
        for constr in constraints_to_remove:
            model.remove(constr)
    return len(constraints_to_remove)

def _is_constraint_redundant(model: gp.Model, constr: gp.Constr) -> bool:
    """Verifica se uma única restrição é redundante."""
    lhs_expr = model.getRow(constr)
    rhs = constr.RHS
    min_activity, max_activity = 0.0, 0.0
    for i in range(lhs_expr.size()):
        var, coeff = lhs_expr.getVar(i), lhs_expr.getCoeff(i)
        if coeff > 0:
            min_activity += coeff * var.LB
            max_activity += coeff * var.UB
        else:
            min_activity += coeff * var.UB
            max_activity += coeff * var.LB
    
    if constr.Sense == GRB.LESS_EQUAL and max_activity <= rhs + 1e-6: return True
    if constr.Sense == GRB.GREATER_EQUAL and min_activity >= rhs - 1e-6: return True
    return False

def _substitute_fixed_variables(model: gp.Model) -> int:
    """
    [CORRIGIDO] Encontra variáveis fixadas (lb=ub), substitui seus valores
    nas restrições e as REMOVE do modelo para evitar loops infinitos.
    Técnica descrita na Seção 4.1 do artigo 'Presolve Reductions in MIP'.
    """
    # Usamos uma list comprehension para encontrar todas as variáveis fixadas de uma vez
    vars_to_substitute = [var for var in model.getVars() if var.LB == var.UB]
    
    if not vars_to_substitute:
        return 0

    print(f"INFO: [Presolve-Fixed] Substituindo e removendo {len(vars_to_substitute)} variáveis fixadas.")

    # Iteramos sobre a lista de variáveis a serem substituídas
    for var in vars_to_substitute:
        fixed_value = var.LB
        
        # Obtém a coluna para ver onde a variável é usada
        col = model.getCol(var)
        
        # Itera sobre as restrições onde a variável aparece
        for i in range(col.size()):
            constr = col.getConstr(i)
            coeff = col.getCoeff(i)
            
            # Remove a variável da restrição (muda o coeficiente para 0)
            model.chgCoeff(constr, var, 0.0)
            
            # Atualiza o RHS da restrição: b' = b - a_j * x_j
            constr.RHS -= coeff * fixed_value
            
        # O PASSO QUE FALTAVA: Remove a variável do modelo por completo
        model.remove(var)

    return len(vars_to_substitute)

def _run_probing(model: gp.Model, original_vars: Dict[str, str]) -> int:
    """
    Executa UMA rodada de Probing em todas as variáveis binárias.
    Técnica descrita na Seção 7.2 do artigo 'Presolve Reductions in MIP'.
    """
    variables_fixed = 0
    
    # Seleciona as variáveis binárias que ainda não estão fixas
    binary_vars_to_probe = [
        var for var in model.getVars() 
        if original_vars.get(var.VarName) == GRB.BINARY and var.LB != var.UB
    ]
    
    print(f"INFO: [Presolve-Probing] Sondando {len(binary_vars_to_probe)} variáveis binárias...")

    for var in binary_vars_to_probe:
        # Se a variável foi fixada por uma sondagem anterior, pule-a
        if var.LB == var.UB:
            continue

        # --- Teste 1: O que acontece se var = 0? ---
        probe_model_0 = model.copy()
        probe_model_0.setParam('OutputFlag', 0)
        probe_var_0 = probe_model_0.getVarByName(var.VarName)
        probe_var_0.UB = 0.0

        # --- ALTERAÇÃO AQUI ---
        # Em vez de apenas propagar bounds, resolvemos o LP para detectar inviabilidade.
        # Isso é um uso permitido do nosso "oráculo LP".
        probe_model_0.optimize() 

        # Agora a checagem de status funcionará corretamente.
        if probe_model_0.Status == GRB.INFEASIBLE:
            print(f"INFO: [Presolve-Probing] Probing em '{var.VarName}=0' provou inviabilidade. Fixando '{var.VarName}=1'.")
            var.LB = 1.0 # Fixa a variável no modelo ORIGINAL
            model.update() # Aplica a fixação para a próxima sondagem
            variables_fixed += 1
            continue

        # --- Teste 2: O que acontece se var = 1? ---
        probe_model_1 = model.copy()
        probe_model_1.setParam('OutputFlag', 0)
        probe_var_1 = probe_model_1.getVarByName(var.VarName)
        probe_var_1.LB = 1.0

        # --- ALTERAÇÃO AQUI ---
        probe_model_1.optimize()

        if probe_model_1.Status == GRB.INFEASIBLE:
            print(f"INFO: [Presolve-Probing] Probing em '{var.VarName}=1' provou inviabilidade. Fixando '{var.VarName}=0'.")
            var.UB = 0.0 # Fixa a variável no modelo ORIGINAL
            model.update() # Aplica a fixação
            variables_fixed += 1

        # --- Teste 2: O que acontece se var = 1? ---
        probe_model_1 = model.copy()
        probe_model_1.setParam('OutputFlag', 0)
        probe_var_1 = probe_model_1.getVarByName(var.VarName)
        probe_var_1.LB = 1.01
        
        _strengthen_bounds(probe_model_1, original_vars)
        
        if probe_model_1.Status == GRB.INFEASIBLE:
            print(f"INFO: [Presolve-Probing] Probing em '{var.VarName}=1' provou inviabilidade. Fixando '{var.VarName}=0'.")
            var.UB = 0.0 # Fixa a variável no modelo original
            variables_fixed += 1
            
    return variables_fixed

# --- O Orquestrador de Presolve ---
def run_presolve(model: gp.Model, original_vars: Dict[str, str]):
    """
    Orquestra todas as rotinas de pré-processamento em um loop de ponto fixo.
    """
    print("-" * 60)
    print("INFO: [Presolve] Iniciando fase de pré-processamento...")
    
    while True:
        model.update()
        total_changes_this_round = 0
        
        changes = _strengthen_bounds(model, original_vars)
        total_changes_this_round += changes
        
        changes = _remove_redundant_constraints(model)
        total_changes_this_round += changes
        
        changes = _substitute_fixed_variables(model)
        total_changes_this_round += changes

        changes = _run_probing(model, original_vars)
        total_changes_this_round += changes

        if total_changes_this_round == 0:
            print("INFO: [Presolve] Ponto fixo atingido. Nenhuma nova redução encontrada.")
            break
    
    model.update() # Aplica todas as últimas alterações
    print("INFO: [Presolve] Fase de pré-processamento concluída.")
    print("-" * 60)