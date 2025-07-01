# CÓDIGO NOVO, FINAL E CORRIGIDO para solver/cuts/gomory.py

import logging
import numpy as np
import math
import gurobipy as gp # Adiciona a importação do gurobipy para usar 'gp.LinExpr'
from typing import List, Dict, Any

from ..problem import Problema

def gerar_cortes_gomory(problema: Problema, solucao_lp: dict, base_info: dict) -> list:
    """
    Gera cortes de Gomory Misto-Inteiros (MIG) a partir de uma solução fracionária de LP.
    Esta versão é robusta para problemas com uma ou múltiplas restrições.
    """
    cortes_gerados = []
    
    try:
        vars_na_base = np.array(base_info['vbasis'])
        slacks_na_base = np.array(base_info['cbasis'])
        # GRB.BASIC é 0. Variáveis na base têm status 0.
        indices_basicos = np.where(np.concatenate((vars_na_base, slacks_na_base)) == 0)[0]

        # --- CORREÇÃO PARA O ValueError ---
        A_matrix = problema.model.getA().toarray()
        
        # Se a matriz A for 1D (problema com 1 restrição), a remodela para 2D
        if A_matrix.ndim == 1:
            A_matrix = A_matrix.reshape(1, -1)

        # Agora o hstack funcionará, pois ambos os arrays são 2D
        A_com_slacks = np.hstack((A_matrix, np.eye(problema.model.NumConstrs)))
        # --- FIM DA CORREÇÃO ---
        
        B_inv = np.linalg.inv(A_com_slacks[:, indices_basicos])

    except (KeyError, np.linalg.LinAlgError, IndexError):
        #logging.warning("Gomory: Informação de base inválida ou matriz singular. Não é possível gerar cortes.")
        return []

    nomes_vars_inteiras = set(problema.variaveis_inteiras_nomes)
    
    # Itera sobre as variáveis básicas para encontrar uma com valor fracionário
    for i, idx_var_basica in enumerate(indices_basicos):
        
        # O corte só se aplica a variáveis de decisão que deveriam ser inteiras
        # e que são menores que o número de variáveis originais (não slacks)
        if idx_var_basica >= len(problema.model.getVars()):
            continue
            
        var_basica = problema.model.getVars()[idx_var_basica]
        if var_basica.VarName not in nomes_vars_inteiras:
            continue
            
        valor_var_basica = solucao_lp.get(var_basica.VarName, 0.0)
        
        if abs(valor_var_basica - round(valor_var_basica)) < 1e-6:
            continue

        f_i = valor_var_basica - math.floor(valor_var_basica)
        if f_i < 1e-6: continue # Evita gerar cortes triviais
        
        linha_tableau = B_inv[i, :] @ A_com_slacks
        
        # Constrói o corte MIG
        # O corte será um dicionário {'coeffs': np.array, 'sentido': str, 'rhs': float}
        # para ser compatível com o cover cut e a nossa interface.
        coeffs_corte = np.zeros(len(problema.model.getVars()))
        
        indices_nao_basicos = np.where(np.concatenate((vars_na_base, slacks_na_base)) != 0)[0]

        for j in indices_nao_basicos:
            # Se a variável não-básica for uma variável de decisão
            if j < len(problema.model.getVars()):
                var_nao_basica = problema.model.getVars()[j]
                var_nome = var_nao_basica.VarName
                a_ij = linha_tableau[j]

                if var_nome in nomes_vars_inteiras:
                    f_j = a_ij - math.floor(a_ij)
                    if f_j <= f_i:
                        coeffs_corte[j] = f_j
                    else:
                        if (1 - f_i) > 1e-6:
                           coeffs_corte[j] = (f_i * (1 - f_j)) / (1 - f_i)
                else: # Variáveis contínuas
                    if a_ij >= 0:
                        coeffs_corte[j] = a_ij
                    else:
                        if (f_i - 1) < -1e-6:
                           coeffs_corte[j] = (f_i / (f_i - 1)) * a_ij
        
        cortes_gerados.append({'coeffs': coeffs_corte, 'sentido': '>=', 'rhs': f_i})
        
        if cortes_gerados:
            logging.info(f"Corte de Gomory Misto-Inteiro gerado a partir da variável '{var_basica.VarName}'.")
            return cortes_gerados

    return cortes_gerados