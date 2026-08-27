* SANKHYA demo instance - a small convex QP.
*
* Illustrative, not a claim about real MRPL blending economics: two intermediate streams,
* LN and HN, must blend to a fixed 100 kbbl/day pool. Running each stream's throughput has
* a strictly convex (quadratic) processing cost - a proxy for the nonlinear energy cost of
* running a compressor or heater away from its design point - so, unlike every other model
* in this demo, the objective cannot be written as a linear function of the variables.
*
* minimize    0.01 LN^2 + 0.02 HN^2
* subject to  LN + HN = 100,           0 <= LN, HN <= 100
*
* QUADOBJ stores Q_ii, and the engine's objective is c'x + 0.5 x'Qx (see
* include/sankhya/model.hpp), so Q_LN,LN = 0.02 and Q_HN,HN = 0.04 give exactly the
* objective above.
*
* KKT, via the multiplier on the equality row: 0.02 LN = 0.04 HN, so LN = 2 HN. Substituting
* into LN + HN = 100 gives HN = 100/3, LN = 200/3. Neither bound is active, so this interior
* stationary point is the optimum: objective = 0.01*(200/3)^2 + 0.02*(100/3)^2 = 200/3, about
* 66.66667. Small enough to check with algebra alone - no numerical library, no SANKHYA code.
NAME          QPBLEND
ROWS
 N  COST
 E  POOL
COLUMNS
    LN        COST         0.0   POOL         1.0
    HN        COST         0.0   POOL         1.0
RHS
    RHS       POOL       100.00
BOUNDS
 UP BND       LN         100.00
 UP BND       HN         100.00
QUADOBJ
    LN        LN           0.02
    HN        HN           0.04
ENDATA
