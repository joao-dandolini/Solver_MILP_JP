import numpy as np

def convert_problem_to_presolver_format(problem):
    """
    Converte um objeto Problem do nosso solver para o formato
    esperado pela classe MIPPresolver do João.

    Args:
        problem: O objeto Problem do nosso solver.

    Returns:
        Uma tupla contendo (dicionário de variáveis, lista de restrições).
    """
    # 1. Traduzir as variáveis
    presolver_variables = {}
    for i, var_name in enumerate(problem.variable_names):
        var_type = 'int' if i in problem.integer_variables else 'continuous'
        presolver_variables[var_name] = {
            'lb': problem.lower_bounds[i],
            'ub': problem.upper_bounds[i],
            'type': var_type
        }

    # 2. Traduzir as restrições
    presolver_constraints = []
    A_dense = problem.constraint_matrix.toarray()
    for i in range(A_dense.shape[0]):
        coeffs_map = {}
        for j, coeff in enumerate(A_dense[i, :]):
            if abs(coeff) > 1e-9: # Apenas coeficientes não-zero
                var_name = problem.variable_names[j]
                coeffs_map[var_name] = coeff
        
        presolver_constraints.append({
            'coeffs': coeffs_map,
            'sense': problem.constraint_senses[i].value,
            'rhs': problem.rhs_vector[i]
        })
        
    return presolver_variables, presolver_constraints