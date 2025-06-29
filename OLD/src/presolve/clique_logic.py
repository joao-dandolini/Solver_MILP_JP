import networkx as nx
from collections import defaultdict
import itertools

def transform_constraint(original_weights, capacity, constraint_type='<='):
    """Transforma a restrição para coeficientes não-negativos e sinaliza variáveis complementadas"""
    transformed_weights = []
    complemented = []
    new_capacity = capacity
    
    # 1) Para cada peso original, se for negativo, fazemos a complementação:
    #    - convertemos para positivo (abs)
    #    - ajustamos a capacidade (novo b aumenta em abs(w))
    #    - marcamos como complementado
    #   Caso contrário, apenas copiamos o valor e marcamos como não complementado.
    for w in original_weights:
        if w < 0:
            transformed_weights.append(abs(w))
            new_capacity += abs(w)
            complemented.append(True)
        else:
            transformed_weights.append(w)
            complemented.append(False)
    
    # 2) Se a restrição era >=, invertemos o sinal de todos os coeficientes
    #    e o sinal da capacidade para manter a forma padrão <=
    if constraint_type == '>=':
        transformed_weights = [-w for w in transformed_weights]
        new_capacity = -new_capacity
        constraint_type = '<='
    
    return transformed_weights, new_capacity, complemented

def detect_cliques(coefficients, capacity, var_names):
    """Implementação do Algoritmo 1 para detecção de cliques"""
    
    # 1) Preparação: ordenar variáveis pelo coeficiente (crescente)
    n = len(coefficients)
    sorted_vars = sorted(enumerate(coefficients), key=lambda x: x[1])
    sorted_indices = [i for i, _ in sorted_vars]
    a = [c for _, c in sorted_vars]                
    
    cliques = []
    
    # 2) Verificação rápida: se os dois maiores coeficientes não violam a capacidade, não há cliques
    if n < 2 or a[-2] + a[-1] <= capacity:
        return cliques
    
    # 3) Encontra o ponto k onde o par (a[k], a[k+1]) já excede a capacidade
    k = next((i for i in range(n-1) if a[i] + a[i+1] > capacity), None)
    if k is None:
        return cliques
    
    # 4) Monta o clique inicial: todos os itens a partir de k
    initial_clique = sorted_indices[k:]
    cliques.append([var_names[i] for i in initial_clique])
    
    # 5) Gera cliques adicionais:
    #    Para cada posição o antes de k, busca o primeiro f ≥ k tal que a[o] + a[f] > capacidade
    #    e forma um novo clique substituindo o “pico” inicial
    for o in range(k-1, -1, -1):
        f = next((i for i in range(k, n) if a[o] + a[i] > capacity), None)
        if f is not None:
            new_clique = [var_names[sorted_indices[o]]] + \
                         [var_names[sorted_indices[i]] for i in range(f, n)]
            cliques.append(new_clique)
    
    return cliques

def build_conflict_graph(all_cliques):
    """Constroi grafo de cliques usando NetrworkX"""
    
    G = nx.Graph()
    for clique in all_cliques:
        for u, v in itertools.combinations(clique, 2):
            G.add_edge(u, v)
    return G

def extend_clique(base_clique, conflict_graph):
    """Implementação do Algoritmo 2 para extensão de clique (índices 1-based)"""
    
    if not base_clique:
        return []
    
    # Começamos com a clique base
    extended = list(base_clique)
    
    # 1) Escolhe o nó d com menor grau na clique (menos conexões = menos restritivo)
    min_degree = float('inf')
    d_node = None
    for node in extended:
        degree = conflict_graph.degree[node]
        if degree < min_degree:
            min_degree = degree
            d_node = node
    if d_node is None:
        return extended
    
    # 2) Lista de candidatos: vizinhos de d que ainda não estão na clique
    L = [n for n in conflict_graph.neighbors(d_node) if n not in extended]
    
    # 3) Extensão gulosa: enquanto existirem candidatos
    while L:
        # 3.1) escolhe l com maior grau (mais conexões potenciais)
        l_node = max(L, key=lambda x: conflict_graph.degree[x])
        L.remove(l_node)
        
        # 3.2) adiciona l_node somente se ele conflitar (for vizinho) com todos os membros atuais
        if all(l_node in conflict_graph.neighbors(member) for member in extended):
            extended.append(l_node)
            # 3.3) novos candidatos: vizinhos de l_node não usados
            for n in conflict_graph.neighbors(l_node):
                if n not in extended and n not in L:
                    L.append(n)
    
    return extended

# Dentro do seu clique_logic.py, substitua esta função:
def clique_cut_separator(lp_solution, conflict_graph):
    """
    Versão HEURÍSTICA do separador de cortes de clique.
    Usa extend_clique em vez de uma busca exaustiva.
    """
    TOLERANCE = 1e-6
    found_cuts = []

    # 1. Pega apenas as variáveis com valor fracionário na solução LP
    fractional_vars = {var for var, val in lp_solution.items() if TOLERANCE < val < 1 - TOLERANCE}
    
    if not fractional_vars:
        return []

    print(f"  DEBUG: Buscando cortes de clique em {len(fractional_vars)} variáveis fracionárias...")

    # 2. Itera sobre todos os pares de variáveis fracionárias que conflitam entre si
    # Estes pares são nossos "cliques semente"
    checked_pairs = set()
    for u in fractional_vars:
        if u not in conflict_graph: continue
        for v in conflict_graph.neighbors(u):
            if v not in fractional_vars: continue
            
            # Garante que não processemos o mesmo par duas vezes (u,v) e (v,u)
            pair = tuple(sorted((u, v)))
            if pair in checked_pairs: continue
            checked_pairs.add(pair)
            
            # 3. Para cada semente, tenta estendê-la para um clique maior
            # Usamos sua função heurística já existente!
            extended_clique = extend_clique(list(pair), conflict_graph)
            
            # 4. Verifica se o clique estendido é violado pela solução atual
            violation = sum(lp_solution.get(node, 0) for node in extended_clique) - 1.0
            
            if violation > TOLERANCE:
                # Se for violado, criamos um novo corte
                cut_coeffs = {var_name: 1.0 for var_name in extended_clique if not var_name.startswith('x̄_')}
                
                # Lógica para variáveis complementadas
                rhs_adjustment = sum(1 for var_name in extended_clique if var_name.startswith('x̄_'))
                for var_name in extended_clique:
                    if var_name.startswith('x̄_'):
                        original_name = var_name.replace('x̄_', '')
                        cut_coeffs[original_name] = -1.0

                cut = {
                    "coeffs": cut_coeffs,
                    "sense": "<=",
                    "rhs": 1.0 - rhs_adjustment
                }
                found_cuts.append(cut)
                # Opcional: parar após encontrar o primeiro corte para acelerar ainda mais
                # return found_cuts 

    return found_cuts