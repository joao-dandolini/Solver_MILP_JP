# test_gomory_source.py
# Cria um problema LP cuja solução ótima é fracionária,
# servindo como uma fonte ideal para um corte de Gomory.

import gurobipy as gp
from gurobipy import GRB

def create_gomory_test_problem():
    
    m = gp.Model("gomory_source_problem")
    m.setParam('OutputFlag', 0)
    
    # Variáveis de decisão inteiras
    x1 = m.addVar(name="x1", vtype=GRB.INTEGER, lb=0)
    x2 = m.addVar(name="x2", vtype=GRB.INTEGER, lb=0)
    
    # Objetivo: Maximizar x2
    m.setObjective(x2, GRB.MAXIMIZE)
    
    # Restrições
    m.addConstr(3 * x1 + 2 * x2 <= 6, "c1")
    m.addConstr(-3 * x1 + 2 * x2 <= 0, "c2")
    
    m.update()
    
    # Salva o problema em um formato que nosso solver principal pode ler
    filename = "gomory_source.lp"
    m.write(filename)
    print(f"Problema fonte para Gomory salvo em '{filename}'")
    
    # Informação extra: Qual é a solução LP ótima?
    # Gurobi nos dirá. Esperamos x2 = 1.5
    lp_sol = m.relax()
    lp_sol.optimize()
    print(f"Solução da relaxação LP (via Gurobi): x1={lp_sol.getVarByName('x1').X:.4f}, x2={lp_sol.getVarByName('x2').X:.4f}")

if __name__ == "__main__":
    create_gomory_test_problem()