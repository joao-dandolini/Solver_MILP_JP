# validar_nosso_parser.py

import logging
import numpy as np
from utils.logger_config import setup_logger
from parsers.mps_parser import ler_arquivo_mps
import sys

# --- Configure o MESMO arquivo que você quer testar ---
caminho_arquivo = "../tests/mas76.mps" 

def print_problema_info(nome: str, problema, mapa_nomes):
    """Função auxiliar para imprimir os detalhes de um problema."""
    mapa_idx_nome = {idx: nome for nome, idx in mapa_nomes.items()}

    print("-" * 50)
    print(f"--- Detalhes do Problema Lido Pelo Nosso Parser: {nome} ---")
    print(f"Sentido do Objetivo: {problema.sentido}")
    print(f"Dimensões de A: {problema.A.shape}")
    print("\nNomes das Variáveis (índice -> nome):")
    for i in range(len(mapa_idx_nome)):
        print(f"  {i}: {mapa_idx_nome[i]}")
    
    print(f"\nVariáveis Inteiras (índices): {problema.variaveis_inteiras}")
    print(f"\nLimites Inferiores (lbs):\n{problema.lbs}")
    print(f"\nLimites Superiores (ubs):\n{problema.ubs}")
    print(f"\nVetor c (Custos):\n{problema.c}")
    print(f"\nVetor b (RHS):\n{problema.b}")
    print("\nMatriz A e Sinais:")
    for i in range(problema.A.shape[0]):
        print(f"  R{i}: {problema.A[i, :]} {problema.sinais[i]} {problema.b[i]}")

    print("-" * 50)

# Em validar_nosso_parser.py

def main():
    setup_logger()
    logging.getLogger().setLevel(logging.CRITICAL) # Desliga quase todos os logs
    print(f"Lendo o arquivo '{caminho_arquivo}' com o nosso parser...")
    
    try:
        problema_lido, mapa_nomes = ler_arquivo_mps(caminho_arquivo, sentido='minimize')
        
        # --- NOVO BLOCO PARA SALVAR EM ARQUIVO ---
        nome_arquivo_saida = "nosso_parser_output.txt"
        print(f"Salvando a saída do nosso parser em '{nome_arquivo_saida}'...")
        
        original_stdout = sys.stdout # Guarda a saída padrão (o console)
        with open(nome_arquivo_saida, 'w', encoding='utf-8') as f:
            sys.stdout = f # Redireciona o print para o arquivo
            print_problema_info(problema_lido.nome, problema_lido, mapa_nomes)
        
        sys.stdout = original_stdout # Restaura a saída padrão para o console
        print("Arquivo salvo com sucesso.")
        # --- FIM DO NOVO BLOCO ---

    except Exception as e:
        print(f"\nERRO DO NOSSO PARSER: {e}")
        import traceback
        traceback.print_exc()


main()