NAME          LOGISTICA_COMPLEXA
OBJSENSE
 MIN
ROWS
 N  COST
 G  DEM_C1
 G  DEM_C2
 L  CAP_W1
 L  CAP_W2
 L  ACT_W1C1
 L  ACT_W1C2
 L  ACT_W2C1
 L  ACT_W2C2
 E  PROD_ESP
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X11       COST                100.0   ACT_W1C1            -9999.0
    X12       COST                120.0   ACT_W1C2            -9999.0
    X21       COST                110.0   ACT_W2C1            -9999.0
    X22       COST                115.0   ACT_W2C2            -9999.0
    P1        COST                15.0    PROD_ESP            1.0
    P2        COST                17.0    PROD_ESP            1.0
    MARK0001  'MARKER'                 'INTEND'
    Q11       COST                2.0     DEM_C1              1.0
    Q11       CAP_W1              1.0     ACT_W1C1            1.0
    Q12       COST                3.0     DEM_C2              1.0
    Q12       CAP_W1              1.0     ACT_W1C2            1.0
    Q21       COST                2.5     DEM_C1              1.0
    Q21       CAP_W2              1.0     ACT_W2C1            1.0
    Q22       COST                2.8     DEM_C2              1.0
    Q22       CAP_W2              1.0     ACT_W2C2            1.0
RHS
    RHS1      DEM_C1              500.0   DEM_C2              700.0
    RHS1      CAP_W1              800.0   CAP_W2              900.0
    RHS1      PROD_ESP            120.0
BOUNDS
 UP BND1      X11                 1.0
 UP BND1      X12                 1.0
 UP BND1      X21                 1.0
 UP BND1      X22                 1.0
 UI BND1      P1                  200.0
 UI BND1      P2                  200.0
ENDATA