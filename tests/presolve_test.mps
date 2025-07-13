* presolve_test.mps
* Este arquivo foi desenhado para testar as rotinas de presolve.
NAME          PRES_TEST
ROWS
 N  OBJ
* R_SINGLE: Uma restrição singleton para testar a substituição
 E  R_SINGLE
* R_GCD: Uma restrição para testar a Redução Euclidiana (MDC)
 E  R_GCD
* R_BOUND: Uma restrição para testar a Propagação de Limites
 L  R_BOUND
* R_TIGHTEN: Uma restrição para testar o Aperto de Coeficientes
 L  R_TIGHTEN

COLUMNS
    X1        OBJ       1.0       R_BOUND   1.0
    X1        R_TIGHTEN 10.0
* X2 é o nosso singleton: X2 = 5
    X2        R_SINGLE  1.0       R_BOUND   1.0
* X3 e X4 para o teste de GCD: 6*X3 + 12*X4 = 18
    X3        OBJ       1.0       R_GCD     6.0
    X4        OBJ       1.0       R_GCD     12.0
* X5 é uma variável binária para o teste de CoefTighten
    X5        OBJ       1.0       R_TIGHTEN 20.0

RHS
    RHSVAL    R_SINGLE  5.0
    RHSVAL    R_GCD     18.0
    RHSVAL    R_BOUND   12.0
    RHSVAL    R_TIGHTEN 28.0

BOUNDS
* X1 tem um limite inicial de [0, 10]
 UP BND1      X1        10.0
* X3 e X4 devem ser inteiras para o teste de GCD
 UI BND1      X3        99.0
 UI BND1      X4        99.0
* X5 é binária para o teste de CoefTighten
 BV BND1      X5

ENDATA