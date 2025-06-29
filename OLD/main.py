from src.utils.parser import parse_file_to_problem
from src.milp_solver.branch_and_bound import MILPSolver
from src.utils.mps_adapter import parse_mps_file

def run():
    '''
    # Teste 1: Problema que força ramificação
    print("==============================================")
    print("   RESOLVENDO: teste_branch.txt")
    print("==============================================")
    problem1 = parse_file_to_problem("data/teste_branch.txt")
    milp_solver1 = MILPSolver(problem1)
    solution1 = milp_solver1.solve()
    print(solution1)

    # Teste 2: Problema com binárias
    print("\n\n==============================================")
    print("   RESOLVENDO: exemplo_simples.txt")
    print("==============================================")
    problem2 = parse_file_to_problem("data/exemplo_simples.txt")
    milp_solver2 = MILPSolver(problem2)
    solution2 = milp_solver2.solve()
    print(solution2)

    # Teste 3: Problema da mochila
    print("\n\n==============================================")
    print("   RESOLVENDO: knapsack_test.txt")
    print("==============================================")
    problem3 = parse_file_to_problem("data/knapsack_test.txt")
    milp_solver3 = MILPSolver(problem3)
    solution3 = milp_solver3.solve()
    print(solution3)

    # Teste 4: Problema complexo
    print("\n\n==============================================")
    print("   RESOLVENDO: desafio_complexo.txt")
    print("==============================================")
    problem4 = parse_file_to_problem("data/desafio_complexo.txt")
    milp_solver4 = MILPSolver(problem4)
    solution4 = milp_solver4.solve()
    print(solution4)

    # Teste 5: Problema da mochila_2
    print("\n\n==============================================")
    print("   RESOLVENDO: knapsack_test_2.txt")
    print("==============================================")
    problem5 = parse_file_to_problem("data/knapsack_test_2.txt")
    milp_solver5 = MILPSolver(problem5)
    solution5 = milp_solver5.solve()
    print(solution5)
    #'''
    #'''
    problem = parse_mps_file("data/instance_0003.mps")
    
    if not problem:
        print("Falha ao carregar o problema. Encerrando.")
        return
    # Passo 2: Criar uma instância do nosso solver principal com o problema carregado.
    milp_solver = MILPSolver(problem)
    
    # Passo 3: Chamar o método para resolver o problema.
    solution = milp_solver.solve()
    
    # Imprime um sumário final, se uma solução foi encontrada
    if solution and solution.get("status") == "Optimal":
        print("\n================================================")
        print(f"  SOLUÇÃO ÓTIMA ENCONTRADA PARA {problem.name}")
        print("================================================")
        print(f"Status: {solution['status']}")
        print(f"Valor da Função Objetivo: {solution['objective_value']:.4f}")
        print("\nValores das Variáveis (não-zero):")
        for var, value in solution["variables"].items():
            if abs(value) > 1e-6:
                print(f"  {var} = {value:.4f}")
    #'''
if __name__ == "__main__":
    run()