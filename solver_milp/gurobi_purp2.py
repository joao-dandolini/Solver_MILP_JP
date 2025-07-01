# teste_knapsack_gurobi_puro.py

import gurobipy as gp
from gurobipy import GRB

def solve_knapsack_com_gurobi_puro():
    """
    Constrói e resolve um problema Knapsack programaticamente usando
    o solver MIP nativo do Gurobi.
    """
    print("Construindo e resolvendo o problema Knapsack programaticamente usando Gurobi...")

    # --- 1. DEFINIÇÃO DOS DADOS DO PROBLEMA ---
    valores = {'x1': 15, 'x2': 12, 'x3': 10, 'x4': 9, 'x5': 7, 'x6': 5, 'x7': 3, 'x8': 1}
    pesos   = {'x1':  8, 'x2':  7, 'x3':  6, 'x4': 5, 'x5': 4, 'x6': 3, 'x7': 2, 'x8': 1}
    capacidade_mochila = 20

    try:
        # --- 2. CRIAÇÃO E CONSTRUÇÃO DO MODELO ---
        
        # O Gurobi irá imprimir seu próprio log, então criamos o ambiente padrão
        env = gp.Env(empty=True)
        env.start()
        modelo = gp.Model("knapsack_gurobi_puro", env=env)

        # Adiciona as variáveis (binárias), já definindo seus custos na função objetivo
        variaveis = modelo.addVars(valores.keys(), vtype=GRB.BINARY, obj=valores, name=list(valores.keys()))
        
        # Adiciona a restrição de capacidade da mochila
        modelo.addConstr(variaveis.prod(pesos) <= capacidade_mochila, "LimiteDePeso")
        
        # Define o sentido da otimização para MAXIMIZAÇÃO
        modelo.modelSense = GRB.MAXIMIZE

        # --- 3. OTIMIZAÇÃO ---
        # Chamamos o solver MIP completo do Gurobi. Ele fará seu próprio B&C.
        modelo.optimize()

        # --- 4. APRESENTAÇÃO DOS RESULTADOS ---
        print("\n" + "="*30)
        print("      RESULTADO DO GUROBI PURO")
        print("="*30)

        if modelo.Status == GRB.OPTIMAL:
            print(f"\nStatus: Solução ótima encontrada!")
            print(f"Valor da Função Objetivo: {modelo.ObjVal:.4f}")
            
            print("\nItens na mochila (variáveis com valor > 0.5):")
            solucao_encontrada = False
            itens_selecionados = []
            for v in modelo.getVars():
                if v.X > 0.5:
                    itens_selecionados.append(v.VarName)
                    solucao_encontrada = True
            
            # Imprime os itens ordenados para fácil comparação
            for item in sorted(itens_selecionados):
                 print(f"  - {item}")
            
            if not solucao_encontrada:
                print("  (Nenhum item selecionado)")

        elif modelo.Status == GRB.INFEASIBLE:
            print("\nStatus: O problema é inviável.")
        elif modelo.Status == GRB.UNBOUNDED:
            print("\nStatus: O problema é ilimitado.")
        else:
            print(f"\nStatus: A otimização terminou com o código de status {modelo.Status}.")

    except gp.GurobiError as e:
        print(f"Erro do Gurobi: {e.message} (código {e.errno})")
    except Exception as e:
        print(f"Um erro inesperado ocorreu: {e}")

# --- Ponto de Entrada do Script ---
if __name__ == "__main__":
    solve_knapsack_com_gurobi_puro()