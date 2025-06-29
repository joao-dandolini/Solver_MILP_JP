import numpy as np
import scipy.sparse as sp
import pulp
from src.core.problem import Problem, ObjectiveSense, ConstraintSense

def parse_mps_file(filepath: str):
    """
    Lê um arquivo .mps usando PuLP para a estrutura geral e uma leitura
    manual para garantir a captura correta da função objetivo.
    """
    print(f"--- Lendo arquivo MPS com PuLP: {filepath} ---")
    
    try:
        # Passo 1: Usar PuLP para a estrutura geral
        variables_dict, lp_problem = pulp.LpProblem.fromMPS(filepath)
        
        variable_names = list(variables_dict.keys())
        var_to_idx = {name: i for i, name in enumerate(variable_names)}
        num_vars = len(variable_names)

        # --- Passo 2: Extração MANUAL e ROBUSTA da Função Objetivo ---
        objective_coeffs = np.zeros(num_vars)
        objective_name = ""
        with open(filepath, 'r') as f:
            in_rows_section = False
            in_cols_section = False
            for line in f:
                parts = line.split()
                if not parts: continue

                if parts[0] == 'ROWS': in_rows_section = True; continue
                if parts[0] == 'COLUMNS': in_rows_section = False; in_cols_section = True; continue
                
                # Encontra o nome da linha de objetivo (tipo 'N')
                if in_rows_section and parts[0] == 'N':
                    objective_name = parts[1]
                    print(f"  Nome da linha de objetivo encontrado: {objective_name}")

                # Extrai os coeficientes da linha de objetivo
                if in_cols_section and len(parts) >= 3 and parts[1] == objective_name:
                    var_name = parts[0]
                    if var_name in var_to_idx:
                        objective_coeffs[var_to_idx[var_name]] = float(parts[2])

        sense = ObjectiveSense.MAXIMIZE if lp_problem.sense == pulp.LpMaximize else ObjectiveSense.MINIMIZE

        # Passo 3: Extrair as restrições, ignorando a linha do objetivo
        constraints_dict = {name: constr for name, constr in lp_problem.constraints.items() if name != objective_name}
        num_constraints = len(constraints_dict)
        A_matrix = sp.lil_matrix((num_constraints, num_vars), dtype=float)
        b_vector = np.zeros(num_constraints)
        senses = []

        for i, (name, constr) in enumerate(constraints_dict.items()):
            for var_obj, coeff in constr.items():
                if var_obj.name in var_to_idx:
                    A_matrix[i, var_to_idx[var_obj.name]] = coeff
            b_vector[i] = constr.constant * -1
            if constr.sense == pulp.LpConstraintLE: senses.append(ConstraintSense.LTE)
            elif constr.sense == pulp.LpConstraintGE: senses.append(ConstraintSense.GTE)
            else: senses.append(ConstraintSense.EQ)

        # O resto da extração permanece o mesmo...
        lower_bounds = np.array([v.lowBound if v.lowBound is not None else 0 for v in variables_dict.values()])
        upper_bounds = np.array([v.upBound if v.upBound is not None else np.inf for v in variables_dict.values()])
        integer_variables = {var_to_idx[v.name] for v in variables_dict.values() if v.cat == pulp.LpInteger or v.cat == pulp.LpBinary}
        
        problem = Problem(
            name=lp_problem.name, objective_sense=sense, objective_coeffs=objective_coeffs,
            constraint_matrix=A_matrix.tocsr(), rhs_vector=b_vector, constraint_senses=senses,
            variable_names=variable_names, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
            integer_variables=integer_variables
        )
        
        print(f"--- Arquivo MPS '{problem.name}' lido e convertido com sucesso! ---")
        return problem

    except Exception as e:
        print(f"Erro fatal ao usar PuLP para ler o arquivo MPS: {e}")
        import traceback
        traceback.print_exc()
        return None