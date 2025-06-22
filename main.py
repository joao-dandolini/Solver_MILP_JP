from src.utils.parser import parse_file_to_problem
from src.milp_solver.branch_and_bound import MILPSolver

def run():
    # Teste 1: Problema que força ramificação
    print("==============================================")
    print("   RESOLVENDO: teste_branch.txt")
    print("==============================================")
    problem1 = parse_file_to_problem("data/teste_branch.txt")
    milp_solver1 = MILPSolver(problem1)
    solution1 = milp_solver1.solve()

    # Teste 2: Problema com binárias
    print("\n\n==============================================")
    print("   RESOLVENDO: exemplo_simples.txt")
    print("==============================================")
    problem2 = parse_file_to_problem("data/exemplo_simples.txt")
    milp_solver2 = MILPSolver(problem2)
    solution2 = milp_solver2.solve()

    # Teste 3: Problema da mochila
    print("\n\n==============================================")
    print("   RESOLVENDO: exemplo_simples.txt")
    print("==============================================")
    problem3 = parse_file_to_problem("data/knapsack_test.txt")
    milp_solver3 = MILPSolver(problem3)
    solution2 = milp_solver3.solve()

if __name__ == "__main__":
    run()