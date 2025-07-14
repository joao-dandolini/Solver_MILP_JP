# main.py
# O ponto de entrada principal para executar o MILP Solver.

import argparse
import sys
from solver import MILPSolver
import traceback

# Defina seu arquivo de teste padrão aqui para facilitar a execução.
# Altere este caminho para o local do seu arquivo .mps de teste principal.
#DEFAULT_PROBLEM_PATH = "./tests/mas76.mps"
#DEFAULT_PROBLEM_PATH = "./tests/instance_0003.mps"
DEFAULT_PROBLEM_PATH = "./tests/Archive/model_S1_Jc0_Js9_T96.mps"

#DEFAULT_PROBLEM_PATH = "./gomory_source.lp"

def main():
    parser = argparse.ArgumentParser(
        description="Um solver de Branch and Bound para problemas MILP.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "problem_file",
        type=str,
        nargs='?',
        default=None,
        help="Caminho para o arquivo do problema no formato .mps."
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="most_infeasible",
        choices=["most_infeasible", "strong", "pseudocost"],
        help="Define a estratégia de branching a ser usada."
    )
    
    parser.add_argument(
        "--use-cuts",
        action="store_true",
        help="Ativa a geração de Cover Cuts no nó raiz."
    )

    parser.add_argument(
        "--use-heuristics",
        action="store_true",
        help="Ativa a heurística Feasibility Pump para encontrar um incumbente inicial."
    )

    parser.add_argument(
        "--use-presolve",
        action="store_true",
        help="Ativa a rotina de pré-processamento (Bound Strengthening)."
    )

    parser.add_argument(
        "--rins-freq",
        type=int,
        default=0, # Padrão 0 significa que a RINS está desligada
        help="Frequência de nós para chamar a heurística RINS (ex: 1000). 0 para desligar."
    )

    parser.add_argument(
        "--dfs-limit",
        type=int,
        default=9999999, # Um padrão alto para que a troca não aconteça a menos que especificado
        help="Número de nós a serem explorados em modo DFS antes de trocar para Best-Bound."
    )

    parser.add_argument(
        "--cut-frequency",
        type=int,
        default=0, # Padrão 0 desliga a geração iterativa
        help="Frequência de nós para tentar gerar cortes (ex: 100). 0 para desligar."
    )
 
    parser.add_argument(
        "--cut-depth",
        type=int,
        default=5, # Padrão razoável
        help="Profundidade máxima na árvore para tentar gerar cortes."
    )

    args = parser.parse_args()
    
    # Lógica para usar um arquivo de teste padrão se nenhum for fornecido
    final_problem_path = args.problem_file
    if final_problem_path is None:
        default_test_file = DEFAULT_PROBLEM_PATH
        print(f"INFO: Nenhum arquivo fornecido. Usando o arquivo de teste padrão: {default_test_file}")
        final_problem_path = default_test_file

    print("--- Iniciando MILP Solver ---")
    
    # --- CORREÇÃO CRUCIAL: Passando o argumento corretamente ---
    solver_config = {
        'branching_strategy': args.strategy,
        'use_cuts': args.use_cuts,
        'use_heuristics': args.use_heuristics,
        'dfs_limit': args.dfs_limit,
        'rins_frequency': args.rins_freq,
        'use_presolve': args.use_presolve,
        'cut_frequency': args.cut_frequency,
        'cut_depth': args.cut_depth 
    }
    solver = MILPSolver(config=solver_config)
    
    try:
        solver.solve(final_problem_path)
    except FileNotFoundError:
        print(f"ERRO: O arquivo '{final_problem_path}' não foi encontrado.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERRO: Ocorreu um erro inesperado durante a execução: {e}", file=sys.stderr)
        # --- ADICIONE ESTA LINHA PARA IMPRIMIR O TRACEBACK DETALHADO ---
        traceback.print_exc()
        # --- FIM DA ADIÇÃO ---
        sys.exit(1)

    print("--- Execução do Solver Concluída ---")

if __name__ == "__main__":
    main()

