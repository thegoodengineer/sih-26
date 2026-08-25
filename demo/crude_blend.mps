* SANKHYA demo instance - crude blending for a diesel pool.
*
* A deliberately small refinery LP that exercises the MPS features most likely to be
* misread: a RANGES entry on a G row, an equality row, and both LO and UP bound types,
* under OBJSENSE MAX. Small enough to check by hand, shaped like the models MRPL runs.
*
* Decision variables, in kbbl/day of crude charged to the CDU:
*   AL  Arab Light   74 $/bbl   diesel yield 0.30   sulphur 1.80 %wt
*   BN  Bonny Light  79 $/bbl   diesel yield 0.45   sulphur 0.14 %wt
*   MU  Murban       77 $/bbl   diesel yield 0.38   sulphur 0.78 %wt
*
* Margin per barrel of crude, with diesel valued at 96 $/bbl and the rest of the barrel at
* 68 $/bbl:   68 + 28 * yield - cost   ->   AL 2.40, BN 1.60, MU 1.64
*
* Rows
*   THRUPUT  CDU throughput, 90 to 120 kbbl/day        (G row widened by RANGES)
*   DIESEL   diesel pool must land exactly on 40       (E row)
*   SULPHUR  blended crude sulphur at most 1.00 %wt,
*            written as 1.80 AL + 0.14 BN + 0.78 MU <= 1.00 (AL + BN + MU)
*            and rearranged to   0.80 AL - 0.86 BN - 0.22 MU <= 0
NAME          CRUDEBLEND
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
    RHS       THRUPUT     90.00   DIESEL      40.00
    RHS       SULPHUR      0.00
RANGES
    RNG       THRUPUT     30.00
BOUNDS
 LO BND       AL          10.00
 UP BND       BN          45.00
 UP BND       MU          60.00
ENDATA
