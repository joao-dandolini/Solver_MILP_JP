from src.utils.parser import parse_file_to_problem
from src.milp_solver.branch_and_bound import MILPSolver

def run():
    # Vamos usar o problema de teste que força a ramificação
    #problem = parse_file_to_problem("data/exemplo_simples.txt")
    problem = parse_file_to_problem("data/teste_branch.txt")
    
    milp_solver = MILPSolver(problem)
    solution = milp_solver.solve()

if __name__ == "__main__":
    run()