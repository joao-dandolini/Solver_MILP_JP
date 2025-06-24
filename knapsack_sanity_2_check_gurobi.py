import gurobipy as gp
from gurobipy import GRB

def solve_mars_payload_challenge():
    """
    Constrói e resolve o problema de otimização da carga útil para Marte
    usando Gurobi.
    """
    try:
        # --- Dados do Problema ---
        equipamentos = [
            "Espectrometro", "Magnetometro", "Perfuratriz", "SensorClima", 
            "BracoRobotico", "LaboratorioQuimico", "SensorUV", "RadarSubsolo", 
            "PainelSolarXL", "AntenaComum", "Termometro", "SensorMetano", 
            "GeradorRTG", "CameraPanoramica", "Microscopio", "ColetorAmostras", 
            "BateriaExtra", "RodaExtra", "ComputadorAvancado", "EscudoTermico"
        ]

        valor_cientifico = {
            "Espectrometro": 50, "Magnetometro": 45, "Perfuratriz": 80, 
            "SensorClima": 35, "BracoRobotico": 60, "LaboratorioQuimico": 95, 
            "SensorUV": 40, "RadarSubsolo": 75, "PainelSolarXL": 110, 
            "AntenaComum": 25, "Termometro": 15, "SensorMetano": 55, 
            "GeradorRTG": 120, "CameraPanoramica": 65, "Microscopio": 30, 
            "ColetorAmostras": 85, "BateriaExtra": 40, "RodaExtra": 20, 
            "ComputadorAvancado": 90, "EscudoTermico": 50
        }

        massa = {
            "Espectrometro": 12, "Magnetometro": 8, "Perfuratriz": 25, "SensorClima": 5,
            "BracoRobotico": 18, "LaboratorioQuimico": 30, "SensorUV": 4, "RadarSubsolo": 22,
            "PainelSolarXL": 35, "AntenaComum": 6, "Termometro": 2, "SensorMetano": 7,
            "GeradorRTG": 40, "CameraPanoramica": 15, "Microscopio": 9, "ColetorAmostras": 28,
            "BateriaExtra": 15, "RodaExtra": 10, "ComputadorAvancado": 13, "EscudoTermico": 16
        }

        volume = {
            "Espectrometro": 15, "Magnetometro": 10, "Perfuratriz": 30, "SensorClima": 5,
            "BracoRobotico": 25, "LaboratorioQuimico": 40, "SensorUV": 3, "RadarSubsolo": 20,
            "PainelSolarXL": 50, "AntenaComum": 8, "Termometro": 2, "SensorMetano": 8,
            "GeradorRTG": 35, "CameraPanoramica": 12, "Microscopio": 7, "ColetorAmostras": 25,
            "BateriaExtra": 10, "RodaExtra": 25, "ComputadorAvancado": 10, "EscudoTermico": 20
        }

        energia = {
            "Espectrometro": 10, "Magnetometro": 8, "Perfuratriz": 20, "SensorClima": 5,
            "BracoRobotico": 15, "LaboratorioQuimico": 25, "SensorUV": 6, "RadarSubsolo": 18,
            "PainelSolarXL": -30, "AntenaComum": 4, "Termometro": 1, "SensorMetano": 9,
            "GeradorRTG": -80, "CameraPanoramica": 12, "Microscopio": 5, "ColetorAmostras": 20,
            "BateriaExtra": 20, "RodaExtra": 2, "ComputadorAvancado": 22, "EscudoTermico": 10
        }

        # --- Criação do Modelo ---
        modelo = gp.Model("DesafioCargaUtilMarte")

        # --- Criação das Variáveis de Decisão ---
        # x[e] será 1 se o equipamento 'e' for escolhido, 0 caso contrário.
        x = modelo.addVars(equipamentos, vtype=GRB.BINARY, name="x")

        # --- Definição da Função Objetivo ---
        modelo.setObjective(x.prod(valor_cientifico), GRB.MAXIMIZE)

        # --- Adição das Restrições ---
        modelo.addConstr(x.prod(massa) <= 150, "Massa_kg")
        modelo.addConstr(x.prod(volume) <= 200, "Volume_m3")
        modelo.addConstr(x.prod(energia) <= 75, "Energia_W")
        
        # Restrição de compatibilidade
        modelo.addConstr(
            x["Perfuratriz"] + x["LaboratorioQuimico"] - x["ColetorAmostras"] <= 1, 
            "Compatibilidade"
        )

        # --- Otimização do Modelo ---
        modelo.optimize()

        # --- Apresentação dos Resultados ---
        if modelo.Status == GRB.OPTIMAL:
            print("-" * 50)
            print("Solução Ótima Encontrada!")
            print(f"Valor Científico Máximo: {modelo.ObjVal}")
            print("\nEquipamentos selecionados para a missão:")
            
            massa_total = 0
            volume_total = 0
            energia_liquida = 0
            
            for e in equipamentos:
                if x[e].X > 0.5:
                    print(f"  - {e}")
                    massa_total += massa[e]
                    volume_total += volume[e]
                    energia_liquida += energia[e]
            
            print("\n--- Resumo dos Recursos Utilizados ---")
            print(f"Massa Total: {massa_total} / 150 kg")
            print(f"Volume Total: {volume_total} / 200 m³")
            print(f"Balanço de Energia: {energia_liquida} / 75 W")
            print("-" * 50)
        else:
            print("Não foi possível encontrar uma solução ótima.")

    except gp.GurobiError as e:
        print(f"Erro no Gurobi, código {e.errno}: {e}")

# Executa a função para resolver o problema
solve_mars_payload_challenge()