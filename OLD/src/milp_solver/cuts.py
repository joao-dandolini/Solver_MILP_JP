import numpy as np

def generate_gomory_cut(tableau, basis, variable_names, problem):
    """
    Gera um único Corte de Gomory a partir de uma tabela ótima do Simplex.

    Args:
        tableau: A matriz numpy da tabela ótima final.
        basis: A lista de índices das variáveis na base.
        variable_names: A lista completa de nomes de variáveis (incluindo folga/excesso).
        problem: O objeto Problem original, para checar quais variáveis são inteiras.

    Returns:
        Um dicionário representando a nova restrição de corte, ou None se nenhum corte for gerado.
    """
    TOLERANCE = 1e-6
    
    # 1. Encontrar uma linha fonte para o corte
    source_row_idx = -1
    for i in range(len(basis)):
        basis_var_idx = basis[i]
        
        # O corte só pode ser gerado a partir de uma variável que deveria ser inteira
        is_integer_var = basis_var_idx < len(problem.variable_names) and basis_var_idx in problem.integer_variables
        
        if is_integer_var:
            value = tableau[i, -1]
            # Verifica se o valor é fracionário
            if abs(value - round(value)) > TOLERANCE:
                source_row_idx = i
                break # Encontramos nossa linha, podemos parar

    if source_row_idx == -1:
        # Nenhuma linha adequada encontrada para gerar um corte
        return None

    print(f"DEBUG: Gerando corte de Gomory a partir da linha {source_row_idx} da tabela.")

    # 2. Derivar o corte da linha fonte
    source_row = tableau[source_row_idx, :]
    
    # Pega a parte fracionária do lado direito (rhs)
    f_rhs = source_row[-1] - np.floor(source_row[-1])
    
    coeffs_map = {}
    # Itera apenas sobre as variáveis NÃO-BÁSICAS
    for col_idx in range(len(variable_names)):
        if col_idx not in basis:
            coeff = source_row[col_idx]
            f_coeff = coeff - np.floor(coeff)
            
            # Apenas coeficientes com parte fracionária são relevantes para o corte
            if f_coeff > TOLERANCE:
                var_name = variable_names[col_idx]
                # Só nos importam as variáveis originais no corte, não as de folga
                if var_name in problem.variable_names:
                    coeffs_map[var_name] = f_coeff

    if not coeffs_map:
        # Nenhum coeficiente útil para o corte
        return None

    # O corte de Gomory é: sum(f_j * x_j) >= f_i
    cut = {
        "coeffs": coeffs_map,
        "sense": ">=",
        "rhs": f_rhs
    }
    
    return cut