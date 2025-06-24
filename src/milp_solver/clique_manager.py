import networkx as nx
import itertools
from src.core.problem import Problem
import numpy as np
from src.presolve.clique_logic import transform_constraint, detect_cliques, build_conflict_graph, clique_cut_separator

TOLERANCE = 1e-6

def initialize_conflict_graph(problem: Problem):
    """
    Função Fachada 1: Analisa a estrutura de um problema e constrói o grafo de conflitos.
    Roda apenas uma vez no início.
    """
    print("  Analisando estrutura do problema para encontrar cliques base...")
    all_cliques = []
    
    # 1. Itera sobre cada restrição para encontrar cliques
    A_matrix = problem.constraint_matrix.toarray()
    for i in range(A_matrix.shape[0]):
        constr_coeffs = A_matrix[i, :]
        var_indices_in_constr = [j for j, coeff in enumerate(constr_coeffs) if abs(coeff) > TOLERANCE]
        
        is_lte = problem.constraint_senses[i].value == '<='
        if not is_lte or not var_indices_in_constr:
            continue
            
        # --- MUDANÇA PRINCIPAL AQUI ---
        # Em vez de uma única verificação, iteramos para sermos mais flexíveis.
        is_candidate_constraint = True
        for j in var_indices_in_constr:
            is_int = j in problem.integer_variables
            lb, ub = problem.lower_bounds[j], problem.upper_bounds[j]

            # Uma variável é válida se for binária [0,1] OU se já estiver fixada em 0 ou 1 pelo presolve.
            is_standard_binary = np.isclose(lb, 0, atol=TOLERANCE) and np.isclose(ub, 1, atol=TOLERANCE)
            is_fixed_at_zero = np.isclose(lb, 0, atol=TOLERANCE) and np.isclose(ub, 0, atol=TOLERANCE)
            is_fixed_at_one = np.isclose(lb, 1, atol=TOLERANCE) and np.isclose(ub, 1, atol=TOLERANCE)

            if not (is_int and (is_standard_binary or is_fixed_at_zero or is_fixed_at_one)):
                # Se encontrarmos UMA variável que não se encaixa, a restrição inteira é desqualificada.
                is_candidate_constraint = False
                break
        
        # Se, após checar todas as variáveis, a restrição ainda for uma candidata, nós prosseguimos.
        if is_candidate_constraint:
        # --- FIM DA MUDANÇA ---
            print(f"  --> Restrição {i} qualificada para análise de clique.")
            # 2. Adapta os dados para o formato das suas funções
            weights = [constr_coeffs[j] for j in var_indices_in_constr]
            capacity = problem.rhs_vector[i]
            var_names_in_constr = [problem.variable_names[j] for j in var_indices_in_constr]

            # 3. Chama suas funções de detecção
            t_weights, t_cap, complemented = transform_constraint(weights, capacity, '<=')
            
            # Constrói os nomes das variáveis, marcando as que foram complementadas
            comp_var_names = []
            final_weights_map = {}
            for k, comp in enumerate(complemented):
                original_name = var_names_in_constr[k]
                transformed_name = f"x̄_{original_name}" if comp else original_name
                comp_var_names.append(transformed_name)
                final_weights_map[transformed_name] = t_weights[k]

            # Filtra apenas as variáveis que ainda estão no problema após a transformação
            valid_coeffs = [final_weights_map[name] for name in comp_var_names]

            cliques = detect_cliques(valid_coeffs, t_cap, comp_var_names)
            if cliques:
                print(f"    --> {len(cliques)} cliques base encontrados.")
                all_cliques.extend(cliques)

    # 4. Constrói e retorna o grafo de conflitos final
    if all_cliques:
        conflict_graph = build_conflict_graph(all_cliques)
        print(f"  Grafo de conflitos construído com {conflict_graph.number_of_nodes()} nós e {conflict_graph.number_of_edges()} arestas.")
        return conflict_graph
    
    print("  Nenhuma estrutura de clique encontrada no problema.")
    return None
    
    print("  Nenhuma estrutura de clique encontrada no problema.")
    return None

def separate_clique_cuts(conflict_graph, lp_solution: dict, problem: Problem):
    """
    Função Fachada 2: Usa o grafo e uma solução LP para gerar cortes de clique violados.
    """
    if not conflict_graph:
        return []

    # Chama seu separador de cortes para encontrar os cliques violados
    violated_cliques = clique_cut_separator(lp_solution, conflict_graph)
    if not violated_cliques:
        return []

    print(f"  DEBUG: Separador encontrou {len(violated_cliques)} cliques violados candidatos.")
    
    generated_cuts = []
    for clique in violated_cliques:
        # A desigualdade de clique é: sum(y_i) <= 1, onde y_i pode ser x_i ou 1-x_i
        # Ex: x1 + x2 + (1-x3) <= 1  --->  x1 + x2 - x3 <= 0
        
        cut_coeffs = {}
        rhs_adjustment = 0
        
        for var_in_clique in clique:
            if var_in_clique.startswith('x̄_'):
                # Para uma variável complementada x̄_j, o termo é (1 - x_j).
                # Na restrição final, o coeficiente de x_j se torna -1 e o RHS diminui em 1.
                original_var_name = var_in_clique.replace('x̄_', '') # Recupera o nome original
                cut_coeffs[original_var_name] = -1.0
                rhs_adjustment += 1
            else:
                # Para uma variável normal x_j, o coeficiente é +1.
                cut_coeffs[var_in_clique] = 1.0
        
        # O RHS da desigualdade de clique é sempre 1.
        # Subtraímos os ajustes das variáveis complementadas.
        final_rhs = 1.0 - rhs_adjustment
        
        cut = {"coeffs": cut_coeffs, "sense": "<=", "rhs": final_rhs}
        generated_cuts.append(cut)

    return generated_cuts