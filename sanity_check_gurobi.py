import gurobipy as gp
from gurobipy import GRB

try:
    # --- Dados do Problema ---
    itens = [
        "lanterna", "comida", "kit_medico", "corda", 
        "camera", "garrafa", "canivete", "bussola"
    ]

    valores = {
        "lanterna": 15, "comida": 30, "kit_medico": 25, "corda": 35,
        "camera": 20, "garrafa": 25, "canivete": 10, "bussola": 12
    }

    pesos = {
        "lanterna": 2, "comida": 4, "kit_medico": 3, "corda": 5,
        "camera": 2, "garrafa": 3, "canivete": 1, "bussola": 1
    }

    capacidade_maxima = 12

    # --- Criação do Modelo ---
    modelo = gp.Model("ProblemaDaMochilaExplorador")

    # --- Criação das Variáveis de Decisão ---
    # x[i] será 1 se o item i for selecionado, e 0 caso contrário.
    x = modelo.addVars(itens, vtype=GRB.BINARY, name="x")

    # --- Definição da Função Objetivo ---
    # Maximizar o valor total dos itens na mochila.
    modelo.setObjective(gp.quicksum(valores[i] * x[i] for i in itens), GRB.MAXIMIZE)

    # --- Adição da Restrição de Capacidade ---
    # O somatório dos pesos dos itens selecionados não pode exceder a capacidade máxima.
    modelo.addConstr(
        gp.quicksum(pesos[i] * x[i] for i in itens) <= capacidade_maxima, "capacidade"
    )

    # --- Otimização do Modelo ---
    modelo.optimize()

    # --- Apresentação dos Resultados ---
    print("-" * 30)
    print(f"Valor ótimo da solução: {modelo.ObjVal}")
    print("Itens a serem levados na mochila:")
    
    peso_total = 0
    for i in itens:
        if x[i].X > 0.5: # Checa se a variável é 1
            print(f"  - {i.replace('_', ' ').capitalize()}")
            peso_total += pesos[i]
            
    print(f"\nPeso total utilizado: {peso_total} / {capacidade_maxima}")
    print("-" * 30)


except gp.GurobiError as e:
    print(f"Erro no Gurobi, código {e.errno}: {e}")

except AttributeError:
    print("Não foi possível encontrar uma solução ótima.")