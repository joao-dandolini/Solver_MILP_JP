# main.py (Pronto para o Desafio de Marte)

import logging
import numpy as np
import gurobipy as gp # Adicione esta importação
from gurobipy import GRB # Adicione esta importação

# Importando todos os nossos módulos
from utils.logger_config import setup_logger
from solver.presolve.aplicador import aplicar_presolve
from solver.milp_solver import MilpSolver
from solver.problem import Problema

def main():
    """
    Ponto de entrada que executa o fluxo completo do solver:
    Leitura -> Presolve -> Resolução (Branch and Bound).
    """
    # 1. CONFIGURAÇÃO
    setup_logger()
    # Para ver mais detalhes do B&B, descomente a linha abaixo
    logging.getLogger().setLevel(logging.DEBUG) 
    
    logging.info("="*20 + " INICIANDO SOLVER MILP COMPLETO " + "="*20)

    # 2. LEITURA DO PROBLEMA
    try:
        # Definimos o caminho do arquivo que queremos resolver
        opcao = 5
        if opcao == 1:
            caminho_problema = "../tests/mas76.mps"
            sentido_desejado = GRB.MINIMIZE
        else:
            caminho_problema = "../tests/instance_0003.mps"
            sentido_desejado = GRB.MAXIMIZE
            
        # Carrega o problema. Gurobi pode assumir o sentido errado.
        problema_original = Problema(problema_input=caminho_problema, nome_problema="meu_problema")

        # Verificamos e corrigimos o sentido do modelo se ele não corresponder ao desejado.
        if problema_original.model.modelSense != sentido_desejado:
            problema_original.model.modelSense = sentido_desejado
            problema_original.model.update()
            sentido_str = "MAXIMIZE" if sentido_desejado == GRB.MAXIMIZE else "MINIMIZE"
            logging.info(f"Sentido da otimização foi corrigido para {sentido_str}.")

    except Exception as e:
        logging.error(f"Falha ao carregar o problema com o Gurobi: {e}")
        return


    # 3. FASE DE PRESOLVE (NOVA LÓGICA)
    problema_para_solver = aplicar_presolve(problema_original)

    # 4. FASE DE RESOLUÇÃO (BRANCH AND BOUND)
    solver = MilpSolver(
        problema=problema_para_solver,
        time_limit_seconds=3000
    )
    solucao_otima, valor_otimo = solver.solve()

    # 5. APRESENTAÇÃO DOS RESULTADOS
    logging.info("="*28 + " RESULTADO FINAL " + "="*28)
    if solucao_otima is not None:
        logging.info(f"Status: SOLUÇÃO ÓTIMA ENCONTRADA")
        # O nome do valor objetivo pode variar, então usamos um termo genérico
        logging.info(f"Valor Objetivo Ótimo: {valor_otimo:.4f}")
        
        # A lógica para obter as variáveis da solução ficou MUITO mais simples.
        # Iteramos diretamente no dicionário da solução.
        variaveis_solucao = [nome for nome, valor in solucao_otima.items() if valor > 0.5]
        
        logging.info(f"Variáveis da solução com valor > 0.5:")
        for item in variaveis_solucao:
            # Imprime o nome da variável e seu valor
            logging.info(f"  - {item} = {solucao_otima[item]}")
    else:
        logging.info("Status: Nenhuma solução inteira viável foi encontrada.")

    logging.info("="*25 + " EXECUÇÃO DO SOLVER CONCLUÍDA " + "="*25)

if __name__ == "__main__":
    main()