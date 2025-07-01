# CÓDIGO NOVO E REFATORADO para solver/cuts/cover.py

import logging
import numpy as np
from typing import Dict, List, Any

# Importa a nova classe Problema
from ..problem import Problema

def gerar_cover_cuts(problema: Problema, solucao_lp: Dict[str, float], indices_candidatos: list) -> List[Dict[str, Any]]:
    """
    Gera Cortes de Cobertura, trabalhando com o novo formato de solução (dicionário).
    """
    cortes_gerados = []
    # Pegamos todas as variáveis do modelo Gurobi, pois precisamos dos nomes
    todas_as_vars_nomes = [v.VarName for v in problema.model.getVars()]
    mapa_vars_idx = {nome: i for i, nome in enumerate(todas_as_vars_nomes)}
    num_vars = len(todas_as_vars_nomes)

    #logging.debug(f"--- Buscando Cover Cuts em {len(indices_candidatos)} restrições candidatas ---")

    for i in indices_candidatos:
        # Pega a restrição do modelo Gurobi
        constr = problema.model.getConstrs()[i]
        rhs = constr.RHS
        
        # Pega os coeficientes e nomes das variáveis desta restrição
        linha = problema.model.getRow(constr)
        
        nomes_na_linha = [linha.getVar(k).VarName for k in range(linha.size())]
        coefs_na_linha = [linha.getCoeff(k) for k in range(linha.size())]

        # Monta a heurística para encontrar o cover
        indices_ordenados = sorted(
            range(len(nomes_na_linha)), 
            key=lambda k: solucao_lp.get(nomes_na_linha[k], 0.0), 
            reverse=True
        )

        cover_nomes, peso_cover = [], 0
        for k in indices_ordenados:
            cover_nomes.append(nomes_na_linha[k])
            peso_cover += coefs_na_linha[k]
            if peso_cover > rhs:
                break
        
        if peso_cover > rhs:
            # Constrói o corte: sum(x_j) <= |C| - 1 para j em C
            soma_lp_cover = sum(solucao_lp.get(nome, 0.0) for nome in cover_nomes)
            limite_do_corte = float(len(cover_nomes) - 1)
            
            if soma_lp_cover > limite_do_corte + 1e-6:
                #logging.info(f"Cover Cut VIOLADO gerado a partir da restrição {i}.")
                
                # Cria o vetor de coeficientes para a nova restrição
                novo_corte_coeffs = np.zeros(num_vars)
                for nome_var in cover_nomes:
                    idx_global = mapa_vars_idx[nome_var]
                    novo_corte_coeffs[idx_global] = 1.0
                    
                cortes_gerados.append({'coeffs': novo_corte_coeffs, 'sentido': '<=', 'rhs': limite_do_corte})
                return cortes_gerados # Retorna o primeiro corte encontrado

    return cortes_gerados