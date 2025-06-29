from src.utils.mps_adapter import parse_mps_file
import numpy as np

def verify_parser():
    """
    Script para testar e verificar a saída do nosso parser de MPS.
    """
    # Usamos um problema conhecido da MIPLIB para o qual sabemos a resposta 
    #problem_filepath = "data/mas76.mps"
    problem_filepath = "data/instance_0003.mps"
    
    print(f"--- Testando o parser com o arquivo: {problem_filepath} ---")
    
    problem = parse_mps_file(problem_filepath)
    
    if not problem:
        print("\nOcorreu um erro e o problema não pôde ser carregado.")
        return

    print("\n--- Verificação do Objeto Problem Criado ---")
    print(f"Nome do Problema: {problem.name}")
    print(f"Sentido do Objetivo: {problem.objective_sense.value}")
    
    # Esta é a verificação mais importante:
    print(f"Número de coeficientes não-zero no objetivo: {np.count_nonzero(problem.objective_coeffs)}")
    print(f"Soma dos coeficientes do objetivo: {np.sum(problem.objective_coeffs):.4f}")
    
    print("\n--- Detalhes da Estrutura ---")
    print(f"Número de Variáveis: {problem.constraint_matrix.shape[1]}")
    print(f"Número de Restrições: {problem.constraint_matrix.shape[0]}")
    print(f"Número de Variáveis Inteiras: {len(problem.integer_variables)}")

if __name__ == "__main__":
    verify_parser()