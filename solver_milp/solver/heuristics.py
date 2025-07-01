# solver/heuristics.py (VERSÃO FINAL COM FEASIBILITY PUMP EXPLÍCITA)

import logging
import numpy as np
from .problem import Problema
from .lp_interface import solve_lp_gurobi
from typing import Tuple, Optional

def heuristica_de_arredondamento(problema: Problema, solucao_lp: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Tenta encontrar uma solução inteira viável arredondando uma solução de LP."""
    if solucao_lp is None:
        return None, None

    logging.debug("Heurística: Tentando arredondamento simples...")
    solucao_arredondada = np.round(solucao_lp)
    
    lhs_calculado = problema.A @ solucao_arredondada
    viavel = True
    for i in range(len(problema.sinais)):
        if (problema.sinais[i] == '<=' and lhs_calculado[i] > problema.b[i] + 1e-6) or \
           (problema.sinais[i] == '>=' and lhs_calculado[i] < problema.b[i] - 1e-6) or \
           (problema.sinais[i] == '=' and abs(lhs_calculado[i] - problema.b[i]) > 1e-6):
            viavel = False
            break
    
    if viavel:
        valor_obj = problema.c @ solucao_arredondada
        logging.info(f"Heurística de arredondamento encontrou solução viável com valor {valor_obj:.4f}.")
        return solucao_arredondada, valor_obj
    else:
        logging.debug("Heurística: Solução arredondada não é viável.")
        return None, None

def heuristica_feasibility_pump(problema: Problema, solucao_lp: np.ndarray, max_iter: int = 10) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Implementa a heurística Feasibility Pump para encontrar uma solução inteira viável."""
    if solucao_lp is None:
        return None, None

    x_k = solucao_lp.copy()
    variaveis_inteiras_idx = problema.variaveis_inteiras

    for k in range(max_iter):
        x_int_k = x_k.copy()
        for j in variaveis_inteiras_idx:
            x_int_k[j] = round(x_k[j])
        
        if _checar_viabilidade_heuristica(problema, x_int_k):
            logging.info(f"Heurística (Pump) encontrou uma solução viável na iteração {k+1}!")
            valor_obj = problema.c @ x_int_k
            return x_int_k, valor_obj
            
        # --- CONSTRUÇÃO DO SUBPROBLEMA DE PROJEÇÃO (MINIMIZAR DISTÂNCIA) ---
        # Objetivo: Encontrar um ponto 'x' no poliedro original que seja o mais próximo de 'x_int_k'.
        # Modelamos isso como: Min Σ |x_j - x_int_k_j| para j nas variáveis inteiras.
        # Isso é um problema de MINIMIZAÇÃO por definição.
        
        # Cria um novo vetor de custos para o problema de projeção
        c_projecao = np.zeros_like(problema.c)
        for j in variaveis_inteiras_idx:
            # Se nossa tentativa é 0 (x_int_k[j]=0), queremos minimizar x_j. Custo = +1.
            # Se nossa tentativa é 1 (x_int_k[j]=1), queremos minimizar (1-x_j), que é o mesmo que maximizar x_j,
            # ou, para um problema de minimização, ter um custo de -1.
            c_projecao[j] = 1.0 if x_int_k[j] < 0.5 else -1.0
        
        problema_projecao = problema.__class__(
            nome="feasibility_pump_proj", A=problema.A, b=problema.b, c=c_projecao,
            sinais=problema.sinais, variaveis_inteiras=problema.variaveis_inteiras,
            lbs=problema.lbs, ubs=problema.ubs, 
            sentido='minimizar' # O objetivo é SEMPRE minimizar a distância.
        )

        status, _, solucao_projetada, _ = solve_lp_gurobi(problema_projecao)
        
        if status != "OPTIMAL":
            logging.warning("Heurística (Pump): LP de projeção falhou.")
            return None, None
        
        x_k = solucao_projetada

    logging.info("Heurística (Pump): Máximo de iterações atingido sem encontrar solução.")
    return None, None

def _checar_viabilidade_heuristica(problema: Problema, solucao: np.ndarray) -> bool:
    """Função auxiliar para verificar se uma solução inteira candidata é viável."""
    lhs_calculado = problema.A @ solucao
    for i in range(len(problema.sinais)):
        if (problema.sinais[i] == '<=' and lhs_calculado[i] > problema.b[i] + 1e-6) or \
           (problema.sinais[i] == '>=' and lhs_calculado[i] < problema.b[i] - 1e-6) or \
           (problema.sinais[i] == '=' and abs(lhs_calculado[i] - problema.b[i]) > 1e-6):
            return False
    return True