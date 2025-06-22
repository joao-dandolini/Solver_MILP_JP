from src.utils.parser import parse_file_to_problem
from src.lp_solver.simplex import SimplexSolver

def run():
    print("--- Iniciando o Solver MILP ---")
    problem_file_path = "data/exemplo_simples.txt"
    problem = parse_file_to_problem(problem_file_path)
    
    print("--- Modelo Matemático Original ---")
    print(problem)
    print("--------------------------------\n")
    
    solver = SimplexSolver(problem)
    solution = solver.solve()
    
    # --- IMPRESSÃO FINAL DOS RESULTADOS ---
    print("\n--- Resumo da Solução (Relaxação LP) ---")
    if solution and solution["status"] == "Optimal":
        print(f"Status: {solution['status']}")
        print(f"Valor da Função Objetivo: {solution['objective_value']:.4f}")
        print("\nValores das Variáveis:")
        for var, value in solution["variables"].items():
            if value > 1e-6: # Só imprime variáveis com valor não-zero
                print(f"  {var} = {value:.4f}")
    else:
        print(f"Não foi possível encontrar uma solução ótima. Status: {solution.get('status', 'Unknown')}")
    print("----------------------------------------")

if __name__ == "__main__":
    run()