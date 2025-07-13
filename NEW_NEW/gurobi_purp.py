import gurobipy as gp

# Use exatamente o mesmo caminho do problema que está falhando.
PROBLEM_FILE = "./tests/instance_0003.mps" 

print(f"--- Verificação de Sanidade com o Gurobi para: {PROBLEM_FILE} ---")

try:
    # Carrega o modelo
    model = gp.read(PROBLEM_FILE)
    
    # Deixa o Gurobi resolver usando todo o seu poder
    model.optimize() 
    
    print("-" * 60)
    # Analisa o resultado final do Gurobi
    if model.Status == gp.GRB.OPTIMAL:
        print(f"RESULTADO: O Gurobi encontrou uma solução ótima!")
        print(f"Valor Objetivo: {model.ObjVal}")
    elif model.Status == gp.GRB.INFEASIBLE:
        print(f"RESULTADO: O Gurobi determinou que o modelo é INFACTÍVEL.")
    elif model.Status == gp.GRB.INF_OR_UNBD:
        print(f"RESULTADO: O Gurobi determinou que o modelo é INFACTÍVEL ou ILIMITADO.")
    else:
        print(f"RESULTADO: O Gurobi terminou com o status: {model.Status} (Consulte a documentação do Gurobi)")

except Exception as e:
    print(f"ERRO ao tentar rodar o Gurobi: {e}")