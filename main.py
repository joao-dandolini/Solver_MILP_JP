from src.utils.parser import parse_file_to_problem
from src.milp_solver.branch_and_bound import MILPSolver # <<<<< NOVA IMPORTAÇÃO

def run():
    problem = parse_file_to_problem("data/exemplo_simples.txt")
    
    # Cria uma instância do nosso solver principal
    milp_solver = MILPSolver(problem)
    
    # Chama o método para resolver o problema MILP
    solution = milp_solver.solve()
    
    # No futuro, imprimiremos a solução final aqui...

if __name__ == "__main__":
    run()