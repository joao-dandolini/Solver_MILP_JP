# presolve.py

import gurobipy as gp
from gurobipy import GRB
import math
from typing import Dict, List

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

def _strengthen_coefficients(model: gp.Model, original_vars: Dict[str, str]) -> int:
    """
    Analisa restrições de mochila (sum(a_j * x_j) <= b) para fixar variáveis.
    Se para uma variável binária x_k, seu coeficiente a_k for maior que o RHS b,
    então x_k deve ser 0.
    
    Esta é uma forma segura e comum de fortalecimento de coeficientes.
    """
    changes = 0
    TOLERANCE = 1e-7
    constrs_to_check = model.getConstrs()

    for constr in constrs_to_check:
        # Foca em restrições do tipo <=
        if constr.Sense != GRB.LESS_EQUAL:
            continue
            
        row = model.getRow(constr)
        rhs = constr.RHS
        
        # Verifica se todos os coeficientes são positivos e as variáveis são binárias
        is_candidate = True
        knapsack_vars = []
        for i in range(row.size()):
            var = row.getVar(i)
            coeff = row.getCoeff(i)
            if original_vars.get(var.VarName) != GRB.BINARY or coeff < 0:
                is_candidate = False
                break
            knapsack_vars.append({'var': var, 'coeff': coeff})
        
        if not is_candidate:
            continue

        # Lógica de fortalecimento principal
        for item in knapsack_vars:
            var_k = item['var']
            coeff_k = item['coeff']
            
            # Se a variável já está fixada em 0, não há o que fazer.
            if var_k.UB == 0.0:
                continue

            # O coeficiente sozinho já viola a restrição.
            if coeff_k > rhs + TOLERANCE:
                print(f"INFO: [Presolve-Coeff] Na restrição '{constr.ConstrName}', coeff({var_k.VarName})={coeff_k} > rhs={rhs}. Fixando {var_k.VarName}=0.")
                var_k.UB = 0.0
                changes += 1
                
    return changes

def _remove_parallel_rows(model: gp.Model) -> int:
    """
    Encontra e remove restrições redundantes (linhas paralelas).
    Ex: a*x <= 10 e a*x <= 12. A segunda é redundante.
    """
    changes = 0
    constrs_to_remove = []
    
    row_signatures = {}
    
    for constr in model.getConstrs():
        row = model.getRow(constr)
        if row.size() == 0:
            constrs_to_remove.append(constr)
            continue
            
        signature_items = []
        for i in range(row.size()):
            signature_items.append((row.getVar(i).VarName, row.getCoeff(i)))
        
        signature_items.sort()
        signature = tuple(signature_items)
        
        # Normaliza o RHS
        normalized_rhs = constr.RHS
        sense = constr.Sense
        
        if signature not in row_signatures:
            row_signatures[signature] = []
        
        row_signatures[signature].append({'constr': constr, 'rhs': normalized_rhs, 'sense': sense})

    for sig, constr_list in row_signatures.items():
        if len(constr_list) < 2:
            continue
            
        # Compara cada par de restrições no grupo
        for i in range(len(constr_list)):
            for j in range(i + 1, len(constr_list)):
                c1_info = constr_list[i]
                c2_info = constr_list[j]
                
                # Foco no caso simples: mesmo sentido
                if c1_info['sense'] == c2_info['sense']:
                    if c1_info['sense'] == GRB.LESS_EQUAL:
                        # Mantém a restrição com o RHS menor (mais apertada)
                        if c1_info['rhs'] <= c2_info['rhs']:
                            constrs_to_remove.append(c2_info['constr'])
                        else:
                            constrs_to_remove.append(c1_info['constr'])
                    elif c1_info['sense'] == GRB.GREATER_EQUAL:
                        # Mantém a restrição com o RHS maior (mais apertada)
                        if c1_info['rhs'] >= c2_info['rhs']:
                            constrs_to_remove.append(c2_info['constr'])
                        else:
                            constrs_to_remove.append(c1_info['constr'])

    # Remove as restrições redundantes de forma segura
    unique_constrs_to_remove = list(set(constrs_to_remove))
    if unique_constrs_to_remove:
        changes = len(unique_constrs_to_remove)
        print(f"INFO: [Presolve-ParallelRows] Removendo {changes} restrições redundantes.")
        for constr in unique_constrs_to_remove:
            # Verifica se a restrição ainda existe antes de remover
            if constr.ConstrName in [c.ConstrName for c in model.getConstrs()]:
                 model.remove(constr)
    
    return changes

def _generate_clique_cuts(model: gp.Model, original_vars: Dict[str, str]) -> int:
    """
    Encontra cliques em um grafo de conflito e adiciona cortes de clique.
    Um conflito x_i + x_j <= 1 forma uma aresta entre os nós i e j.
    Um clique no grafo é um conjunto de variáveis mutuamente exclusivas.
    """
    changes = 0
    
    # 1. Construir o grafo de conflitos
    adj = {v.VarName: [] for v in model.getVars() if original_vars.get(v.VarName) == GRB.BINARY}
    binary_vars = set(adj.keys())
    
    for constr in model.getConstrs():
        if constr.Sense == GRB.LESS_EQUAL and constr.RHS == 1.0:
            row = model.getRow(constr)
            if row.size() == 2:
                var1, var2 = row.getVar(0), row.getVar(1)
                coeff1, coeff2 = row.getCoeff(0), row.getCoeff(1)
                
                if var1.VarName in binary_vars and var2.VarName in binary_vars and \
                   abs(coeff1 - 1.0) < 1e-7 and abs(coeff2 - 1.0) < 1e-7:
                    adj[var1.VarName].append(var2.VarName)
                    adj[var2.VarName].append(var1.VarName)

    # 2. Encontrar cliques maximais (usando o algoritmo de Bron-Kerbosch, uma versão simples)
    added_cliques = set()

    for var_name in binary_vars:
        # Tenta construir um clique começando com var_name e seus vizinhos
        potential_clique = {var_name}
        # Pega vizinhos que também são vizinhos entre si
        for neighbor in adj[var_name]:
            is_clique_member = True
            for member in potential_clique:
                if member != var_name and neighbor not in adj[member]:
                    is_clique_member = False
                    break
            if is_clique_member:
                potential_clique.add(neighbor)
        
        if len(potential_clique) > 2:
            # Ordena para criar uma assinatura única e evitar duplicatas
            clique_signature = tuple(sorted(list(potential_clique)))
            if clique_signature not in added_cliques:
                print(f"INFO: [Presolve-Cliques] Encontrado clique de tamanho {len(clique_signature)}. Adicionando corte.")
                model.addConstr(gp.quicksum(model.getVarByName(v) for v in clique_signature) <= 1)
                added_cliques.add(clique_signature)
                changes += 1
                
    return changes

def run_presolve(model: gp.Model, original_vars: Dict[str, str]) -> str:
    """Orquestra as rotinas de presolve em um loop até que não haja mais mudanças."""
    print("-" * 60)
    print("INFO: [Presolve] Iniciando fase de pré-processamento avançado...")
    
    MAX_PRESOLVE_ROUNDS = 2
    total_changes_made = 0

    for i in range(MAX_PRESOLVE_ROUNDS):
        print(f"INFO: [Presolve] Passada {i+1}...")
        changes_in_round = 0
        
        # 1. Simplificações baratas
        changes = _fix_from_singletons(model)
        if changes == -1: return 'INFEASIBLE'
        changes_in_round += changes
        
        changes = _remove_parallel_rows(model)
        changes_in_round += changes
        
        # 2. Propagação de bounds (ainda relativamente barata)
        changes = _propagate_bounds(model, original_vars)
        if changes == -1: return 'INFEASIBLE'
        changes_in_round += changes
        
        # 3. Técnicas de fortalecimento
        changes = _strengthen_coefficients(model, original_vars)
        changes_in_round += changes
        
        changes = _generate_clique_cuts(model, original_vars)
        changes_in_round += changes
        
        # 4. Probing (geralmente a mais cara)
        changes = _probe_binary_variables(model, original_vars)
        if changes == -1: return 'INFEASIBLE'
        changes_in_round += changes
        
        model.update()
        
        print(f"INFO: [Presolve] Mudanças nesta passada: {changes_in_round}")
        total_changes_made += changes_in_round
        if changes_in_round == 0:
            print("INFO: [Presolve] Nenhuma nova melhoria encontrada. Encerrando pré-processamento.")
            break
    
    print(f"INFO: [Presolve] Fase de pré-processamento concluída. Total de mudanças: {total_changes_made}.")
    return 'SUCCESS'
