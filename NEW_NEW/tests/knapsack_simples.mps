* NOME DO PROBLEMA
NAME          KNAPSACK_EX

* =================================================================
* Problema da Mochila (Knapsack)
* OBJETIVO: Maximizar o valor total dos itens.
* RESTRICAO: O peso total dos itens não pode passar de 20.
* VARIAVEIS: 8 itens, cada um pode ser escolhido (1) ou não (0).
* =================================================================

ROWS
 N  VALOR_TOT
 L  PESO_TOT
COLUMNS
    x1      VALOR_TOT     15.0     PESO_TOT       8.0
    x2      VALOR_TOT     12.0     PESO_TOT       7.0
    x3      VALOR_TOT     10.0     PESO_TOT       6.0
    x4      VALOR_TOT      8.0     PESO_TOT       5.0
    x5      VALOR_TOT      7.0     PESO_TOT       4.0
    x6      VALOR_TOT      5.0     PESO_TOT       3.0
    x7      VALOR_TOT      3.0     PESO_TOT       2.0
    x8      VALOR_TOT      1.0     PESO_TOT       1.0
RHS
    RHS1      PESO_TOT      20.0
BOUNDS
 UP BND1      x1         1.0
 UP BND1      x2         1.0
 UP BND1      x3         1.0
 UP BND1      x4         1.0
 UP BND1      x5         1.0
 UP BND1      x6         1.0
 UP BND1      x7         1.0
 UP BND1      x8         1.0
ENDATA