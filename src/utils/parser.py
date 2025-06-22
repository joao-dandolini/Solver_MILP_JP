# Arquivo: src/utils/parser.py (Versão Final)

import re
import numpy as np
from scipy.sparse import csr_matrix
from src.core.problem import Problem, ObjectiveSense, ConstraintSense

def parse_file_to_problem(file_path: str) -> Problem:
    """
    Lê um arquivo de texto com a definição de um problema MILP
    e o converte em um objeto Problem.
    """
    # --- ESTRUTURAS DE DADOS TEMPORÁRIAS ---
    current_section = None
    problem_name = ""
    objective_sense = None
    obj_coeffs_map = {}
    constraints_list = []
    bounds_map = {}
    integer_vars_set = set()
    binary_vars_set = set()

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1].upper()
            
            elif current_section == 'NAME':
                problem_name = line
            
            elif current_section == 'OBJECTIVE':
                sense_str, expr = line.split(':', 1)
                sense_map = {"max": ObjectiveSense.MAXIMIZE, "maximize": ObjectiveSense.MAXIMIZE, "min": ObjectiveSense.MINIMIZE, "minimize": ObjectiveSense.MINIMIZE}
                objective_sense = sense_map.get(sense_str.strip().lower())
                if objective_sense is None: raise ValueError(f"Sentido da função objetivo '{sense_str}' não reconhecido.")
                
                terms = re.findall(r'([+\-]?\s*\d+\.?\d*)\s*(\w+\d*)', expr)
                for coeff_str, var_name in terms:
                    obj_coeffs_map[var_name] = float(coeff_str.replace(" ", ""))

            elif current_section == 'CONSTRAINTS':
                parts = line.split(':', 1); constraint_name, expr = (parts[0].strip(), parts[1]) if len(parts) == 2 else ("", parts[0])
                sense_str = next((s for s in ['<=', '>=', '=='] if s in expr), None)
                if not sense_str: raise ValueError(f"Sentido de restrição não encontrado em: {line}")
                
                lhs_expr, rhs_str = expr.split(sense_str); rhs_val = float(rhs_str.strip())
                constraint_sense = ConstraintSense(sense_str)
                
                lhs_coeffs = {var: float(c.replace(" ", "")) for c, var in re.findall(r'([+\-]?\s*\d+\.?\d*)\s*(\w+\d*)', lhs_expr)}
                constraints_list.append({"name": constraint_name, "coeffs": lhs_coeffs, "sense": constraint_sense, "rhs": rhs_val})

            # --- LÓGICA PARA BOUNDS E INTEGERS ---
            elif current_section == 'BOUNDS':
                parts = re.split(r'(>=|<=)', line)
                var_name = parts[0].strip()
                sense = parts[1]
                value = float(parts[2].strip())
                
                if var_name not in bounds_map: bounds_map[var_name] = {'lower': -np.inf, 'upper': np.inf}
                if sense == '>=': bounds_map[var_name]['lower'] = value
                elif sense == '<=': bounds_map[var_name]['upper'] = value

            elif current_section == 'INTEGERS':
                integer_vars_set.add(line.strip())

            elif current_section == 'BINARY':
                binary_vars_set.add(line.strip())

    # --- MONTAGEM FINAL DO OBJETO PROBLEM ---
    # 1. Obter lista ordenada de todas as variáveis
    all_vars = sorted(list(
        obj_coeffs_map.keys() | 
        {k for c in constraints_list for k in c['coeffs'].keys()} | 
        bounds_map.keys() | 
        integer_vars_set | 
        binary_vars_set  # Inclui as binárias na lista de todas as variáveis
    ))
    var_to_idx = {var: i for i, var in enumerate(all_vars)}
    num_vars = len(all_vars)

    # 2. Construir vetor de custos 'c'
    c = np.array([obj_coeffs_map.get(var, 0) for var in all_vars])

    # 3. Construir matriz 'A', vetor 'b' e sentidos das restrições
    num_constraints = len(constraints_list)
    b = np.zeros(num_constraints)
    constraint_senses = []
    rows, cols, data = [], [], []
    for i, const in enumerate(constraints_list):
        b[i] = const['rhs']
        constraint_senses.append(const['sense'])
        for var, coeff in const['coeffs'].items():
            rows.append(i); cols.append(var_to_idx[var]); data.append(coeff)
    A = csr_matrix((data, (rows, cols)), shape=(num_constraints, num_vars))

    # 4. Construir vetores de bounds e conjuntos de vars inteiras/binárias
    lower_bounds = np.zeros(num_vars)
    upper_bounds = np.full(num_vars, np.inf)

    # Aplicar bounds explícitos primeiro
    for var, bounds in bounds_map.items():
        idx = var_to_idx[var]
        lower_bounds[idx] = bounds.get('lower', 0)
        upper_bounds[idx] = bounds.get('upper', np.inf)
    
    # <<<<<<< NOVA LÓGICA DE SOBRESCRITA PARA BINÁRIAS >>>>>>>
    # Sobrescrever os limites para variáveis binárias, garantindo que sejam [0, 1]
    for var in binary_vars_set:
        idx = var_to_idx[var]
        lower_bounds[idx] = 0
        upper_bounds[idx] = 1

    # Unir variáveis inteiras e binárias para a restrição de integralidade
    integer_indices = {var_to_idx[var] for var in integer_vars_set | binary_vars_set}
    
    # 5. Criar e retornar o objeto
    return Problem(
        name=problem_name,
        objective_sense=objective_sense,
        objective_coeffs=c,
        constraint_matrix=A,
        rhs_vector=b,
        constraint_senses=constraint_senses,
        variable_names=all_vars,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        integer_variables=integer_indices
    )