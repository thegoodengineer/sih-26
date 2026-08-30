* SANKHYA demo instance - a small convex MIQP (integer variables AND a quadratic objective).
*
* The same two-stream blend as demo/qp_blend.mps, with one change that moves it into a
* different problem class: throughput is now scheduled in WHOLE units rather than
* continuously. A refinery does not run a stream at 66.667 kbbl/day because the arithmetic
* says so - it runs an integer number of batches, tanks or campaign days - and the moment
* that is written down the problem is no longer a QP.
*
* minimize    0.01 LN^2 + 0.02 HN^2
* subject to  LN + HN = 100,   0 <= LN, HN <= 100,   LN and HN INTEGER
*
* HAND CHECK, by substitution - no solver, no numerical library:
*
*   The equality leaves one degree of freedom. Put LN = t, so HN = 100 - t, and
*
*     f(t) = 0.01 t^2 + 0.02 (100 - t)^2
*          = 0.01 t^2 + 0.02 (10000 - 200 t + t^2)
*          = 0.03 t^2 - 4 t + 200
*
*   which is a parabola minimised at t = 4 / 0.06 = 200/3 = 66.667, giving 200/3 = 66.66667.
*   That is the CONTINUOUS optimum, and it is not an integer, so the integer optimum is one
*   of its two neighbours:
*
*     f(66) = 0.03 (4356) - 264 + 200 = 130.68 - 64 = 66.68
*     f(67) = 0.03 (4489) - 268 + 200 = 134.67 - 68 = 66.67   <-- optimum
*
*   So the answer is LN = 67, HN = 33, objective 66.67, and the relaxation it came from is
*   66.66667. Those two numbers DIFFER, which is the point of the instance: a solver that
*   quietly dropped integrality would report 66.66667 and look entirely correct doing it.
*   Here the difference is visible at the fifth decimal, and the integer answer is provably
*   the better of exactly two candidates.
*
* QUADOBJ stores the lower triangle of Q and the objective is c'x + 0.5 x'Qx, so Q_ii of
* 0.02 and 0.04 give the 0.01 and 0.02 coefficients above - the same convention, and the
* same values, as demo/qp_blend.mps.
NAME          MIQPBLND
ROWS
 N  COST
 E  POOL
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    LN        COST         0.0   POOL         1.0
    HN        COST         0.0   POOL         1.0
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       POOL       100.00
BOUNDS
 UP BND       LN         100.00
 UP BND       HN         100.00
QUADOBJ
    LN        LN           0.02
    HN        HN           0.04
ENDATA
