* SANKHYA case study - product distribution from refineries to depots.
*
* PS26119 names 'transportation and supply chain management'. This is the classical
* balanced transportation problem, chosen because of a property it has BY
* CONSTRUCTION rather than by accident:
*
*   TOTAL SUPPLY EQUALS TOTAL DEMAND, so the seven equality rows are linearly
*   DEPENDENT - summing the three supply rows and summing the four demand rows give
*   the same equation. The constraint matrix has rank 6, not 7.
*
* That redundancy is why this instance is here. A simplex implementation that assumes
* its basis matrix is nonsingular, or an LU that cannot handle a structurally singular
* column, fails on it. Every vertex is also degenerate: a basis needs m + n - 1 = 6
* basic variables but the tableau has 7 rows, so at least one basic variable sits at
* zero at every iteration and ratio-test ties are the norm rather than the exception.
* This is the standard cycling test bed.
*
*   min  sum_ij f_ij x_ij
*   s.t. sum_j x_ij = supply_i    for each plant
*        sum_i x_ij = demand_j    for each depot
*
*   supply [300.0, 400.0, 500.0] = 1200
*   demand [250.0, 350.0, 400.0, 200.0] = 1200
*
*   freight cost per unit, plant (row) to depot (column):
*     MANGALORE  [4.0, 6.0, 9.0, 5.0]
*     KOCHI      [5.0, 3.0, 7.0, 8.0]
*     CHENNAI    [6.0, 7.0, 4.0, 3.0]
NAME          SUPPLYCH
ROWS
 N  FREIGHT
 E  SUP1
 E  SUP2
 E  SUP3
 E  DEM1
 E  DEM2
 E  DEM3
 E  DEM4
COLUMNS
    X11       FREIGHT                4  SUP1                   1
    X11       DEM1                   1
    X12       FREIGHT                6  SUP1                   1
    X12       DEM2                   1
    X13       FREIGHT                9  SUP1                   1
    X13       DEM3                   1
    X14       FREIGHT                5  SUP1                   1
    X14       DEM4                   1
    X21       FREIGHT                5  SUP2                   1
    X21       DEM1                   1
    X22       FREIGHT                3  SUP2                   1
    X22       DEM2                   1
    X23       FREIGHT                7  SUP2                   1
    X23       DEM3                   1
    X24       FREIGHT                8  SUP2                   1
    X24       DEM4                   1
    X31       FREIGHT                6  SUP3                   1
    X31       DEM1                   1
    X32       FREIGHT                7  SUP3                   1
    X32       DEM2                   1
    X33       FREIGHT                4  SUP3                   1
    X33       DEM3                   1
    X34       FREIGHT                3  SUP3                   1
    X34       DEM4                   1
RHS
    RHS       SUP1                 300  SUP2                 400
    RHS       SUP3                 500  DEM1                 250
    RHS       DEM2                 350  DEM3                 400
    RHS       DEM4                 200
ENDATA
