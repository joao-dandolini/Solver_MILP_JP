# cut_generator.py
# Contém a lógica para gerar planos de corte para o solver.

from typing import Dict, List, Tuple, Any
import gurobipy as gp
from gurobipy import GRB

def find_cover_cuts(
    model: gp.Model, 
    lp_solution: Dict[str, float],
    original_vars: Dict[str, str],
    debug: bool = False
) -> List[Tuple[List[str], int]]:
    """
    [ATUALIZADO] Tenta encontrar e gerar TODOS os Cover Cuts possíveis em uma única passagem.
    Retorna os componentes do corte (lista de nomes de vars, lado direito) em vez de uma string.

    Args:
        model: O modelo Gurobi atual.
        lp_solution: A solução fracionária do LP do nó atual.
        original_vars: Um dicionário mapeando nomes de variáveis para seus tipos originais (B, I, C).
        debug: Se True, imprime informações detalhadas do processo.

    Returns:
        Uma lista de tuplas. Cada tupla contém:
        - Uma lista de nomes de variáveis que formam o lado esquerdo do corte.
        - Um inteiro que é o lado direito do corte.
    """
    if debug: print("--- Buscando Cover Cuts ---")
    
    generated_cuts_data = []
    constr_candidates = 0
    TOLERANCE = 1e-6

    # Percorre todas as restrições do modelo para encontrar candidatas a mochila
    for i, constr in enumerate(model.getConstrs()):

        if constr.ConstrName.startswith("cover_cut"):
            continue
        
        # Permite restrições de '<=' e também de '=='
        if constr.Sense not in [GRB.LESS_EQUAL, GRB.EQUAL]:
            continue
        
        lhs = model.getRow(constr)
        is_knapsack_candidate = True
        knapsack_vars = []

        # Verifica se todos os termos se encaixam no padrão
        for j in range(lhs.size()):
            var = lhs.getVar(j)
            coeff = lhs.getCoeff(j)

            # Para ser um cover cut clássico, as variáveis devem ser binárias e os coeficientes positivos
            if original_vars.get(var.VarName) != GRB.BINARY or coeff < TOLERANCE:
                is_knapsack_candidate = False
                break
            
            knapsack_vars.append({'var': var, 'coeff': coeff})

        if not is_knapsack_candidate:
            continue
            
        constr_candidates += 1
        if debug: print(f"\n[DEBUG] Restrição Candidata #{i} encontrada.")

        # Filtra variáveis que já estão em zero na solução LP
        active_vars = [kv for kv in knapsack_vars if lp_solution.get(kv['var'].VarName, 0) > TOLERANCE]
        
        # Ordena pela sua importância na solução LP (maior valor primeiro)
        active_vars.sort(key=lambda kv: lp_solution.get(kv['var'].VarName, 0), reverse=True)
        if debug: print(f"[DEBUG]   - Vars ativas na restrição: {[item['var'].VarName for item in active_vars]}")

        cover_set = []
        cover_weight = 0.0
        rhs_limit = constr.RHS
        if debug: print(f"[DEBUG]   - Limite da restrição (RHS): {rhs_limit}")

        cover_found = False
        for item in active_vars:
            cover_set.append(item['var'])
            cover_weight += item['coeff']
            if debug: print(f"[DEBUG]     - Adicionando '{item['var'].VarName}', peso atual do cover: {cover_weight:.2f}")

            if cover_weight > rhs_limit:
                rhs_cut = len(cover_set) - 1
                cut_data = (list(cover_set), rhs_cut) # Usamos list() para copiar
                generated_cuts_data.append(cut_data)
                
                # Log para o usuário ver
                lhs_display = " + ".join([v.VarName for v in cover_set])
                print(f"INFO: Cover Cut Gerado a partir da restrição {i}: {lhs_display} <= {rhs_cut}")
                
                cover_found = True
                break # Sai para a próxima restrição
        
        if not cover_found and debug:
            print(f"[DEBUG]   - FALHA: Não foi possível encontrar um cover para a restrição {i}.")
    
    if debug: print(f"\n--- Fim da Busca. Candidatas a mochila: {constr_candidates}. Cortes gerados: {len(generated_cuts_data)} ---")
    return generated_cuts_data