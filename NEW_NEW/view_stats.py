# view_stats.py
# Um script simples para ler e exibir os resultados do cProfile.

import pstats
from pstats import SortKey

# Carrega o arquivo de estatísticas gerado pelo cProfile
stats = pstats.Stats('profile_dificil.pstats')

# Limpa os nomes dos arquivos para facilitar a leitura
stats.strip_dirs()

# Ordena as estatísticas pela métrica mais importante: 'tottime'
# 'tottime' é o tempo total gasto na função, excluindo o tempo de sub-chamadas.
stats.sort_stats(SortKey.TIME)

print("="*80)
print("RELATÓRIO DE PERFORMANCE DO SOLVER")
print("Ordenado por 'tottime' - tempo gasto DENTRO da função.")
print("="*80)

# Imprime as 15 funções mais 'pesadas'
stats.print_stats(15)

# Você também pode querer ver o tempo cumulativo ('cumtime')
# stats.sort_stats(SortKey.CUMULATIVE)
# print("\n" + "="*80)
# print("RELATÓRIO DE PERFORMANCE DO SOLVER")
# print("Ordenado por 'cumtime' - tempo gasto na função E em suas sub-chamadas.")
# print("="*80)
# stats.print_stats(15)