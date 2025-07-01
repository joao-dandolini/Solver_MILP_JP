# teste_gurobi_puro.py

import gurobipy as gp
from gurobipy import GRB
import logging

# --- Configuração ---
# Caminho para o arquivo MPS. Ajuste se o seu estiver em outro lugar.
#caminho_arquivo = "../tests/mas76.mps" 
caminho_arquivo = "../tests/instance_0003.mps"
        
print(f"Lendo e resolvendo o arquivo: {caminho_arquivo} usando Gurobi...")

try:
    # 1. Criar o ambiente e ler o modelo diretamente do arquivo
    # A função gp.read() é o parser nativo do Gurobi.
    env = gp.Env(empty=True)
    # O Gurobi irá imprimir seu próprio log, então não desligamos o OutputFlag
    env.start()
    modelo = gp.read(caminho_arquivo, env=env)

    # 2. Otimizar o modelo
    # O método .optimize() do Gurobi executa seu próprio algoritmo de B&C
    modelo.optimize()

    # 3. Apresentar os resultados
    print("\n" + "="*30)
    print("      RESULTADO DO GUROBI")
    print("="*30)

    if modelo.Status == GRB.OPTIMAL:
        print(f"\nStatus: Solução ótima encontrada!")
        print(f"Valor da Função Objetivo: {modelo.ObjVal:.4f}")
        
        print("\nVariáveis da solução com valor > 0.5:")
        solucao_encontrada = False
        for v in modelo.getVars():
            if v.X > 0.5:
                print(f"  - {v.VarName}: {v.X}")
                solucao_encontrada = True
        
        if not solucao_encontrada:
            print("  (Nenhuma variável com valor > 0.5 na solução ótima)")

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