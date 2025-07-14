# cut_generator.py
# Contém a lógica para gerar planos de corte para o solver.

# cut_generator.py (Versão Final, Universal e Multi-Corte)

from typing import Dict, List, Tuple, Any
import gurobipy as gp
from gurobipy import GRB, quicksum
import math
import numpy as np
from simplex import SimplexSolver

# Em cut_generator.py

def _convert_to_standard_form(model: gp.Model) -> Tuple:
    """
    Converte um modelo Gurobi para a forma padrão (MAX, <=) para nosso Simplex.
    Retorna os dados numéricos (A, b, c) e a lista de variáveis na ordem correta.
    """
    model.update()
    
    gurobi_vars = model.getVars()
    var_names_in_order = [v.VarName for v in gurobi_vars]
    
    # Mapeia o nome de cada variável para sua posição (índice da coluna) na matriz A
    var_name_to_idx = {v.VarName: i for i, v in enumerate(gurobi_vars)}
    
    # 1. Lida com o Objetivo
    if model.ModelSense == GRB.MINIMIZE:
        c = np.array([-v.Obj for v in gurobi_vars])
    else:
        c = np.array([v.Obj for v in gurobi_vars])

    # 2. Lida com as Restrições
    A_rows = []
    b = []
    
    for constr in model.getConstrs():
        lhs = model.getRow(constr)
        
        # --- LÓGICA CORRIGIDA PARA CONSTRUIR A LINHA DA MATRIZ A ---
        row_coeffs = np.zeros(len(gurobi_vars))
        for i in range(lhs.size()):
            var = lhs.getVar(i)
            coeff = lhs.getCoeff(i)
            col_idx = var_name_to_idx[var.VarName]
            row_coeffs[col_idx] = coeff
        # --- FIM DA LÓGICA CORRIGIDA ---

        rhs = constr.RHS
        
        if constr.Sense == GRB.GREATER_EQUAL:
            A_rows.append(-row_coeffs)
            b.append(-rhs)
        elif constr.Sense == GRB.LESS_EQUAL:
            A_rows.append(row_coeffs)
            b.append(rhs)
        elif constr.Sense == GRB.EQUAL:
            A_rows.append(row_coeffs)
            b.append(rhs)
            A_rows.append(-row_coeffs)
            b.append(-rhs)
            
    A = np.array(A_rows)
    b = np.array(b)
    
    return A, b, c, gurobi_vars, var_names_in_order

def find_gomory_cuts(
    model: gp.Model,
    original_vars_map: Dict[str, str],
    max_cuts_per_round: int = 5
) -> List[Tuple[Dict[str, float], Any, float]]:
    
    print("INFO: [Gomory] Iniciando busca inteligente por cortes de Gomory...")
    
    # 1. Converte o problema para a forma padrão (MAX, <=)
    A, b, c, gurobi_vars, var_names_in_order = _convert_to_standard_form(model)
    
    # 2. Resolve o LP com nosso SimplexSolver
    simplex_solver = SimplexSolver(list(c), A.tolist(), list(b))
    result = simplex_solver.solve()
    
    if result is None:
        return []

    # 3. Encontra TODAS as fontes de corte potenciais
    potential_cut_sources = []
    for row_idx, basic_var_idx in enumerate(result.basic_variables_indices):
        value = result.tableau[row_idx, -1]
        
        if basic_var_idx < result.num_vars:
            var_name = var_names_in_order[basic_var_idx]
            if original_vars_map.get(var_name) != GRB.CONTINUOUS and abs(value - round(value)) > 1e-6:
                fractional_part = abs(value - math.floor(value))
                depth_score = 0.5 - abs(fractional_part - 0.5)
                potential_cut_sources.append({'row_idx': row_idx, 'var_name': var_name, 'value': value, 'score': depth_score})

    if not potential_cut_sources:
        return []

    # 4. Classifica as fontes pela "profundidade" (as melhores primeiro)
    potential_cut_sources.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"INFO: [Gomory] {len(potential_cut_sources)} fontes de corte encontradas. Selecionando as {max_cuts_per_round} melhores.")

    generated_cuts_recipes = []
    
    # 5. Gera cortes a partir das melhores fontes, até o limite
    for source in potential_cut_sources[:max_cuts_per_round]:
        print(f"INFO: [Gomory] Gerando corte a partir da fonte '{source['var_name']}' (valor: {source['value']:.4f}, score: {source['score']:.4f})")
        
        row_idx = source['row_idx']
        value = source['value']
        
        fractional_part_rhs = value - math.floor(value)
        cut_coeffs = {}

        for col_idx in range(result.num_vars + result.num_constraints):
            if col_idx not in result.basic_variables_indices:
                coeff = result.tableau[row_idx, col_idx]
                f_j = coeff - math.floor(coeff)
                
                if f_j > 1e-6:
                    if col_idx < result.num_vars:
                        non_basic_var_name = var_names_in_order[col_idx]
                        cut_coeffs[non_basic_var_name] = cut_coeffs.get(non_basic_var_name, 0.0) + f_j
                    else:
                        constr_idx = col_idx - result.num_vars
                        fractional_part_rhs -= f_j * b[constr_idx]
                        for var_k_idx, var_k_name in enumerate(var_names_in_order):
                            cut_coeffs[var_k_name] = cut_coeffs.get(var_k_name, 0.0) - f_j * A[constr_idx, var_k_idx]
        
        # --- FILTRO DE QUALIDADE NUMÉRICA ---
        # Verificamos se o corte gerado é forte o suficiente para ser útil.
        max_abs_coeff = 0.0
        if cut_coeffs:
            max_abs_coeff = max(abs(c) for c in cut_coeffs.values())

        # Se o maior coeficiente no corte for muito pequeno, o corte é "poeira"
        # e provavelmente será ignorado pelo Gurobi.
        MIN_COEFF_TOLERANCE = 1e-7
        if max_abs_coeff < MIN_COEFF_TOLERANCE:
            print(f"INFO: [Gomory] Corte da fonte '{source['var_name']}' foi descartado por ser numericamente fraco.")
            continue  # Pula para a próxima fonte de corte potencial

        # Se o corte passar no teste, nós criamos a receita e a adicionamos.
        final_cut_recipe = (cut_coeffs, GRB.GREATER_EQUAL, fractional_part_rhs)
        print(f"INFO: [Gomory] Receita de corte VÁLIDA gerada.")
        generated_cuts_recipes.append(final_cut_recipe)
        # --- FIM DO FILTRO ---
            
    return generated_cuts_recipes

# Em cut_generator.py

def find_cover_cuts(
    model: gp.Model, 
    lp_solution: Dict[str, float],
    original_vars: Dict[str, str]
) -> List[Tuple[List[str], int]]:
    """
    Tenta encontrar e gerar Cover Cuts a partir de restrições de mochila 0-1.
    """
    print("INFO: [Cover Cut] Iniciando busca por Cover Cuts...")
    
    generated_cuts_data = []
    TOLERANCE = 1e-6

    for constr in model.getConstrs():
        if constr.ConstrName.startswith(("cover_", "gomory_")):
            continue
        
        if constr.Sense not in [GRB.LESS_EQUAL, GRB.EQUAL]:
            continue
        
        lhs = model.getRow(constr)
        is_knapsack_candidate = True
        knapsack_vars = []

        for j in range(lhs.size()):
            var = lhs.getVar(j)
            coeff = lhs.getCoeff(j)
            if original_vars.get(var.VarName) != GRB.BINARY or coeff < TOLERANCE:
                is_knapsack_candidate = False
                break
            knapsack_vars.append({'var_name': var.VarName, 'coeff': coeff})

        if not is_knapsack_candidate:
            continue
        
        # --- LOG: ENCONTRAMOS UMA CANDIDATA ---
        print(f"  -> [Cover Cut] Restrição '{constr.ConstrName}' é uma candidata à mochila.")

        active_vars = [kv for kv in knapsack_vars if lp_solution.get(kv['var_name'], 0) > TOLERANCE]
        active_vars.sort(key=lambda kv: lp_solution.get(kv['var_name'], 0), reverse=True)
        
        cover_set_names = []
        cover_weight = 0.0
        rhs_limit = constr.RHS

        for item in active_vars:
            cover_set_names.append(item['var_name'])
            cover_weight += item['coeff']
            
            if cover_weight > rhs_limit:
                rhs_cut = len(cover_set_names) - 1
                
                # Verifica se o corte é violado pela solução LP atual
                lhs_cut_value = sum(lp_solution.get(v_name, 0.0) for v_name in cover_set_names)
                if lhs_cut_value > rhs_cut + TOLERANCE:
                    # --- LOG: ENCONTRAMOS UM CORTE VÁLIDO ---
                    cut_display = " + ".join(cover_set_names)
                    print(f"    => [Cover Cut] CORTE VIOLADO ENCONTRADO: {cut_display} <= {rhs_cut}")
                    generated_cuts_data.append((cover_set_names, rhs_cut))
                
                # Paramos após encontrar o primeiro cover minimal para esta restrição
                break 
    
    print(f"INFO: [Cover Cut] Busca finalizada. {len(generated_cuts_data)} novas receitas de Cover Cut geradas.")
    return generated_cuts_data
