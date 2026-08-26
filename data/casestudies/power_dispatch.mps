* SANKHYA case study - power system dispatch (single-period unit commitment).
*
* PS26119 names 'power system dispatch' in scope. This is its smallest honest form:
* each generating unit has a marginal cost, a minimum stable generation level it
* cannot run below, a maximum, and a start-up cost paid only if it is committed.
* The min-stable-generation level is what makes this a MILP rather than an LP - a
* unit is either off, or on and producing AT LEAST its minimum.
*
*   min  sum_g ( c_g * p_g + s_g * u_g )
*   s.t. sum_g p_g = D                     (DEMAND, met exactly)
*        sum_g Pmax_g * u_g >= 1.15 * D    (RESERVE, spinning reserve margin)
*        p_g - Pmax_g * u_g <= 0           (CAPMX, off means zero)
*        p_g - Pmin_g * u_g >= 0           (CAPMN, on means at least Pmin)
*        u_g binary
*
*   D = 250 MW, reserve margin 1.15
*
*   unit   $/MWh   Pmin   Pmax   start-up
*   GA     10.0   20.0  100.0      100.0
*   GB     12.0   30.0  120.0       80.0
*   GC     20.0   10.0  150.0       50.0
*   GD      8.0   50.0   60.0      450.0
*
* GD is the trap, and the margin is deliberately thin. It burns the cheapest fuel on
* the system, so a merit-order rule commits it on sight. Doing so displaces 60 MW - 40
* from GB at 12 and 20 from GC at 20 - which saves 40*12 + 20*20 = 880 in fuel, against
* 60*8 + 450 = 930 to run and start it. Committing GD is therefore worse by exactly 50
* out of 3270, about 1.5%. At a start-up cost of 400 instead of 450 the two commitments
* TIE at 3270 while using completely different units, which is how this instance was
* found: the solver and the exhaustive oracle returned the same cost and disagreed on
* every unit. Alternate optima are normal in dispatch models and are the reason the
* oracle here compares the objective, not the assignment.
NAME          PWRDISP
ROWS
 N  COST
 E  DEMAND
 G  RESERVE
 L  CAPMXGA
 G  CAPMNGA
 L  CAPMXGB
 G  CAPMNGB
 L  CAPMXGC
 G  CAPMNGC
 L  CAPMXGD
 G  CAPMNGD
COLUMNS
    PGA       COST                  10  DEMAND                 1
    PGA       CAPMXGA                1  CAPMNGA                1
    PGB       COST                  12  DEMAND                 1
    PGB       CAPMXGB                1  CAPMNGB                1
    PGC       COST                  20  DEMAND                 1
    PGC       CAPMXGC                1  CAPMNGC                1
    PGD       COST                   8  DEMAND                 1
    PGD       CAPMXGD                1  CAPMNGD                1
    MARKER0000  'MARKER'                 'INTORG'
    UGA       COST                 100  RESERVE              100
    UGA       CAPMXGA             -100  CAPMNGA              -20
    UGB       COST                  80  RESERVE              120
    UGB       CAPMXGB             -120  CAPMNGB              -30
    UGC       COST                  50  RESERVE              150
    UGC       CAPMXGC             -150  CAPMNGC              -10
    UGD       COST                 450  RESERVE               60
    UGD       CAPMXGD              -60  CAPMNGD              -50
    MARKER0001  'MARKER'                 'INTEND'
RHS
    RHS       DEMAND               250  RESERVE            287.5
BOUNDS
 UP BND       PGA                  100
 UP BND       PGB                  120
 UP BND       PGC                  150
 UP BND       PGD                   60
 BV BND       UGA
 BV BND       UGB
 BV BND       UGC
 BV BND       UGD
ENDATA
