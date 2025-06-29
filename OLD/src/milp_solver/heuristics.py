import numpy as np

TOLERANCE = 1e-6

def rounding_heuristic(problem, initial_lp_solution):
    """
    Tenta encontrar uma solução inteira inicial arredondando a solução do LP.

    Args:
        problem: O objeto Problem original.
        initial_lp_solution: O dicionário da solução da relaxação LP do nó raiz.

    Returns:
        Um dicionário de solução inteira se uma for encontrada, senão None.
    """
    print("  Executando Heurística de Arredondamento...")
    rounded_vars = {}
    
    # 1. Arredonda todas as variáveis (inteiras ou não) para o inteiro mais próximo
    for var_name, value in initial_lp_solution["variables"].items():
        rounded_vars[var_name] = round(value)

    # 2. Verifica a viabilidade da solução arredondada
    # Checa cada restrição do problema original
    A = problem.constraint_matrix.toarray()
    b = problem.rhs_vector
    
    # Monta o vetor da solução arredondada na ordem correta
    x_rounded = np.array([rounded_vars.get(name, 0) for name in problem.variable_names])
    
    # Calcula o lado esquerdo de todas as restrições de uma vez: Ax
    lhs_values = A @ x_rounded
    
    for i, sense in enumerate(problem.constraint_senses):
        lhs = lhs_values[i]
        rhs = b[i]
        
        is_violated = False
        if sense.value == "<=" and lhs > rhs + TOLERANCE:
            is_violated = True
        elif sense.value == ">=" and lhs < rhs - TOLERANCE:
            is_violated = True
        elif sense.value == "==" and abs(lhs - rhs) > TOLERANCE:
            is_violated = True
            
        if is_violated:
            print("  --> Solução arredondada é inviável. Heurística falhou.")
            return None

    # 3. Se chegou até aqui, a solução é viável!
    print("  --> Heurística encontrou uma solução inteira viável!")
    
    # Calcula o valor objetivo da solução heurística
    heuristic_obj_val = problem.objective_coeffs @ x_rounded
    
    return {
        "status": "Heuristic Solution",
        "variables": rounded_vars,
        "objective_value": heuristic_obj_val
    }