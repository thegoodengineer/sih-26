* SANKHYA demo instance - the crude blend, over-constrained on purpose (#217).
*
* The same diesel-pool blend as demo/crude_blend.mps with one number changed: the pool
* commitment is 60 kbbl/day instead of 40. The CDU window tops out at 120 kbbl/day and the
* best diesel yield of the three crudes is 0.45, so the most diesel the unit can make is
* 0.45 * 120 = 54. The commitment cannot be met, and a planner needs to be told WHICH two
* lines of the model fight each other - THRUPUT and DIESEL - not just that there is no
* answer. The sulphur limit and the crude bounds play no part and must not be named.
*
*   AL  Arab Light   diesel yield 0.30   sulphur 1.80 %wt
*   BN  Bonny Light  diesel yield 0.45   sulphur 0.14 %wt
*   MU  Murban       diesel yield 0.38   sulphur 0.78 %wt
NAME          CRUDEBLEND_INFEASIBLE
OBJSENSE
    MAXIMIZE
ROWS
 N  MARGIN
 G  THRUPUT
 E  DIESEL
 L  SULPHUR
COLUMNS
    AL        MARGIN       2.40   THRUPUT      1.00
    AL        DIESEL       0.30   SULPHUR      0.80
    BN        MARGIN       1.60   THRUPUT      1.00
    BN        DIESEL       0.45   SULPHUR     -0.86
    MU        MARGIN       1.64   THRUPUT      1.00
    MU        DIESEL       0.38   SULPHUR     -0.22
RHS
    RHS       THRUPUT     90.00   DIESEL      60.00
    RHS       SULPHUR      0.00
RANGES
    RNG       THRUPUT     30.00
BOUNDS
 LO BND       AL          10.00
 UP BND       BN          45.00
 UP BND       MU          60.00
ENDATA
