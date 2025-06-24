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

def clique_cut_separator(lp_solution, conflict_graph, min_viol=0.0, max_calls=1000):
    """Implementação do Algoritmo 3 usando extend_clique para clique_extension"""
    
    # 1) Identifica variáveis fracionárias e constrói o subgrafo induzido
    fractional_vars = [var for var, val in lp_solution.items() if 0 < val < 1]
    print(f"Fractional variables: {fractional_vars}")
    subgraph = defaultdict(set)
    for var in fractional_vars:
        #subgraph[var] = {n for n in conflict_graph[var] if n in fractional_vars}
        subgraph[var] = {n for n in conflict_graph[var] if n in fractional_vars} if var in conflict_graph else set()
    
    # 2) Atribui pesos a cada variável (usa x ou 1–x para complementadas)
    weights = {}
    for var in fractional_vars:
        if var.startswith('x̄'):
            orig = 'x' + var[2:]
            weights[var] = 1 - lp_solution.get(orig, 0)
        else:
            weights[var] = lp_solution[var]
    print("Variable weights:")
    for var, w in weights.items():
        print(f"{var}: {w:.2f}")
    
    # 3) Busca cliques com peso ≥ 1 + min_viol via Bron–Kerbosch com pivot
    min_weight = 1.0 + min_viol
    print(f"\nSearching for cliques with weight ≥ {min_weight:.2f}")
    
    def bronkkerbosch(R, P, X):
        nonlocal calls, found_cliques
        calls += 1
        if calls > max_calls:
            return
        if not P and not X:
            total = sum(weights[v] for v in R)
            if total >= min_weight - 1e-6:
                found_cliques.append(list(R))
            return
        # escolhe pivô para reduzir branching
        u = max(P | X, key=lambda x: weights[x], default=None)
        for v in list(P - subgraph.get(u, set())):
            bronkkerbosch(R | {v}, P & subgraph[v], X & subgraph[v])
            P.remove(v)
            X.add(v)
    
    calls, found_cliques = 0, []
    bronkkerbosch(set(), set(subgraph.keys()), set())
    print(f"\nFound {len(found_cliques)} cliques after {calls} recursive calls:")
    for i, clique in enumerate(found_cliques):
        total = sum(weights[v] for v in clique)
        print(f"Clique {i+1}: {sorted(clique)} (weight: {total:.2f})")
    
    # 4) Estende cada clique encontrada usando extend_clique
    print("\nExtending cliques with extend_clique:")
    extended_cliques = []
    for clique in found_cliques:
        print(f"\nExtending base clique: {clique}")
        ext = extend_clique(clique, conflict_graph)
        extended_cliques.append(ext)
        print(f"Extended clique: {ext}")
    
    return extended_cliques
