# CÓDIGO NOVO E PROGRAMÁTICO para main.py

import logging
import gurobipy as gp
from gurobipy import GRB

from utils.logger_config import setup_logger
from solver.milp_solver import MilpSolver
from solver.problem import Problema

def main():
    setup_logger()
    logging.info("="*20 + " INICIANDO SOLVER MILP (MODO PROGRAMÁTICO) " + "="*20)

    # --- 1. CONSTRUÇÃO DO PROBLEMA KNAPSACK DIRETAMENTE NO CÓDIGO ---
    
    # Dados do problema
    valores = {'x1': 15, 'x2': 12, 'x3': 10, 'x4': 9, 'x5': 7, 'x6': 5, 'x7': 3, 'x8': 1}
    pesos   = {'x1':  8, 'x2':  7, 'x3':  6, 'x4': 5, 'x5': 4, 'x6': 3, 'x7': 2, 'x8': 1}
    capacidade_mochila = 20
    
    # Cria um modelo Gurobi vazio
    m = gp.Model("knapsack_programatico")

    # Adiciona as variáveis ao modelo
    # Usamos m.addVars para criar todas de uma vez, já associando seus custos no objetivo
    variaveis = m.addVars(valores.keys(), vtype=GRB.BINARY, obj=valores, name=list(valores.keys()))
    
    # Adiciona a restrição de capacidade
    m.addConstr(variaveis.prod(pesos) <= capacidade_mochila, "LimiteDePeso")
    
    # Define o sentido da otimização
    m.modelSense = GRB.MAXIMIZE
    m.update() # Atualiza o modelo com as novas informações

    try:
        # --- 2. RESOLUÇÃO ---
        # Passamos o objeto modelo 'm' do Gurobi diretamente para a nossa classe Problema
        problema_knapsack = Problema(problema_input=m, nome_problema="Knapsack")

        solver = MilpSolver(
            problema=problema_knapsack,
            time_limit_seconds=60 
        )
        solucao_otima, valor_otimo = solver.solve()

        # --- 3. APRESENTAÇÃO DOS RESULTADOS ---
        logging.info("="*28 + " RESULTADO FINAL " + "="*28)
        if solucao_otima:
            logging.info(f"Status: Solução encontrada!")
            logging.info(f"Valor Objetivo Ótimo: {valor_otimo:.4f}")
            variaveis_solucao = [nome for nome, valor in solucao_otima.items() if valor > 0.5]
            logging.info(f"Itens na mochila:")
            for item in sorted(variaveis_solucao):
                logging.info(f"  - {item}")
        else:
            logging.info("Status: Nenhuma solução encontrada.")

    except Exception as e:
        logging.error(f"Ocorreu um erro durante a execução: {e}", exc_info=True)

if __name__ == "__main__":
    main()