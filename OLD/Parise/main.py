# main.py
import multiprocessing as mp
from factory_location_problem import create_factory_location_problem
from presolver import Presolver, InfeasibleProblemError
from work_stealing_solver import WorkStealingSolver
from problem_parser import create_problem_from_file

def main():
    """
    Ponto de entrada principal para configurar e resolver o problema de MIP.
    """
    #problem = create_factory_location_problem(
    #    num_fabricas=50, 
    #    num_clientes=200, 
    #    seed=123
    #)
    
    problem = create_problem_from_file('knapsack_test_2.txt')

    try:
        print("\nIniciando a fase de Presolve...")
        # Para problemas grandes, desativar o probing pode ser mais rápido
        presolver = Presolver(problem, use_probing=False)
        simplified_problem = presolver.presolve()
        
        solver = WorkStealingSolver(
            simplified_problem, 
            num_workers=None, # None usa todos os cores disponíveis
            timeout=300,   # Timeout em segundos (ex: 5 minutos).
            stagnation_limit=5000, # Para após 5.000 nós sem melhoria.
            mip_gap_tolerance=0.00000000000001 # Para quando o gap for menor que 0.1%
        )
        solver.solve()
        
    except InfeasibleProblemError as e:
        print(f"\nO PROCESSO FOI ENCERRADO PELO PRESOLVE.")
        print(f"RAZÃO: O problema foi provado como inviável. Detalhes: {e}")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    mp.freeze_support()
    main()