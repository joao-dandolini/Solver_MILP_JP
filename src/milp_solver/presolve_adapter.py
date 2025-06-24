from src.core.problem import Problem, ObjectiveSense, ConstraintSense
from scipy.sparse import csr_matrix
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

def convert_presolver_to_problem_format(presolver_vars, presolve_constrs, original_problem):
    """
    Converte os dados de um MIPPresolver de volta para um objeto Problem.

    Args:
        presolver_vars: O dicionário de variáveis do presolver (possivelmente modificado).
        presolve_constrs: A lista de restrições do presolver (possivelmente modificada).
        original_problem: O objeto Problem original, para pegar dados que não mudam (nome, objetivo).

    Returns:
        Um novo objeto Problem, simplificado.
    """
    
    # 1. Obter a lista final de variáveis e criar um mapa para os índices
    variable_names = sorted(list(presolver_vars.keys()))
    var_to_idx = {name: i for i, name in enumerate(variable_names)}
    num_vars = len(variable_names)

    # 2. Construir os atributos das variáveis (limites, tipo, custos)
    lower_bounds = np.zeros(num_vars)
    upper_bounds = np.zeros(num_vars)
    integer_variables = set()
    objective_coeffs = np.zeros(num_vars)

    for i, var_name in enumerate(variable_names):
        info = presolver_vars[var_name]
        lower_bounds[i] = info['lb']
        upper_bounds[i] = info['ub']
        if info['type'] == 'int':
            integer_variables.add(i)
        
        # O presolve não muda os custos, então pegamos do problema original
        if var_name in original_problem.variable_names:
            original_idx = original_problem.variable_names.index(var_name)
            objective_coeffs[i] = original_problem.objective_coeffs[original_idx]

    # 3. Construir a matriz de restrições A e o vetor b
    rows, cols, data = [], [], []
    rhs_vector = []
    constraint_senses = []

    for i, constr in enumerate(presolve_constrs):
        rhs_vector.append(constr['rhs'])
        constraint_senses.append(ConstraintSense(constr['sense']))
        for var_name, coeff in constr['coeffs'].items():
            if var_name in var_to_idx: # A variável pode ter sido eliminada
                rows.append(i)
                cols.append(var_to_idx[var_name])
                data.append(coeff)
    
    constraint_matrix = csr_matrix((data, (rows, cols)), shape=(len(presolve_constrs), num_vars))

    # 4. Criar e retornar o novo objeto Problem simplificado
    simplified_problem = Problem(
        name=f"{original_problem.name}_presolved",
        objective_sense=original_problem.objective_sense,
        objective_coeffs=objective_coeffs,
        constraint_matrix=constraint_matrix,
        rhs_vector=np.array(rhs_vector),
        constraint_senses=constraint_senses,
        variable_names=variable_names,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        integer_variables=integer_variables
    )
    
    return simplified_problem