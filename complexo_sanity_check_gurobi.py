import gurobipy as gp
from gurobipy import GRB

# Dados do Problema
produtos = ["Pa", "Pb", "Pc"]

lucros = {
    "Pa": 18,
    "Pb": 25,
    "Pc": 12
}

# Coeficientes das restrições
maquinario = {"Pa": 7, "Pb": 11, "Pc": 5}
mao_de_obra = {"Pa": 10, "Pb": 8, "Pc": 13}
material_raro = {"Pa": 3, "Pb": 2, "Pc": 4}

# Limites (lado direito) das restrições
limite_maquinario = 150
limite_mao_de_obra = 200
limite_material_raro = 70
minimo_contrato = 10

try:
    # --- Criação do Modelo ---
    modelo = gp.Model("DesafioPlanejamentoProducao")

    # --- Criação das Variáveis de Decisão ---
    # x[p] será a quantidade a ser produzida do produto p.
    # vtype=GRB.INTEGER garante que as variáveis sejam inteiras.
    x = modelo.addVars(produtos, vtype=GRB.INTEGER, name="x")

    # --- Definição da Função Objetivo ---
    # Maximizar o lucro total.
    modelo.setObjective(
        gp.quicksum(lucros[p] * x[p] for p in produtos), 
        GRB.MAXIMIZE
    )

    # --- Adição das Restrições ---
    modelo.addConstr(
        gp.quicksum(maquinario[p] * x[p] for p in produtos) <= limite_maquinario, 
        "Maquinario"
    )

    modelo.addConstr(
        gp.quicksum(mao_de_obra[p] * x[p] for p in produtos) <= limite_mao_de_obra, 
        "MaoDeObra"
    )

    modelo.addConstr(
        gp.quicksum(material_raro[p] * x[p] for p in produtos) <= limite_material_raro, 
        "MaterialRaro"
    )
    
    modelo.addConstr(
        x["Pa"] + x["Pb"] >= minimo_contrato, 
        "Contrato"
    )

    # --- Otimização do Modelo ---
    modelo.optimize()

    # --- Apresentação dos Resultados ---
    print("-" * 40)
    print(f"Solução ótima encontrada:")
    print(f"Lucro Máximo Total: {modelo.ObjVal:.2f}")
    print("\nQuantidade a ser produzida de cada produto:")
    
    for p in produtos:
        if x[p].X > 0: # Mostra apenas os produtos com produção
            print(f"  - Produto {p}: {int(x[p].X)} unidades")
            
    print("-" * 40)


except gp.GurobiError as e:
    print(f"Erro no Gurobi, código {e.errno}: {e}")

except AttributeError:
    # Este erro ocorre se o Gurobi não encontrar uma solução viável.
    print("Não foi possível encontrar uma solução ótima para o problema.")