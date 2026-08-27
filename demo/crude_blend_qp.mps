* SANKHYA demo instance - crude blending with price impact, as a convex QP.
*
* The SAME physical blending decision as demo/crude_blend.mps - same three crudes, same
* CDU throughput window, same diesel commitment, same sulphur specification - with the one
* change that turns a refinery blending LP into a QP: the price of a crude is not constant
* in the volume purchased.
*
* WHY THIS IS QUADRATIC. A blending LP assumes each crude has a fixed delivered price, so
* the hundredth barrel costs what the first one did. For a refiner lifting a material
* fraction of a grade's available supply that is false: bidding for more moves the price
* against you. Model the marginal delivered price as rising linearly with the volume taken,
*
*     price_j(x_j) = base_j + p_j * x_j
*
* and the total spend on crude j is the integral of that, not the product:
*
*     spend_j = (base_j + 0.5 * p_j * x_j) * x_j = base_j * x_j + 0.5 * p_j * x_j^2
*
* so the margin loses a term 0.5 * p_j * x_j^2. In the QPS convention - objective written
* as c'x + 0.5 x'Qx, with only the lower triangle of Q listed - that is exactly Q_jj = -p_j
* with the linear margins c unchanged from the LP. The 0.5 the file convention carries and
* the 0.5 in the integral above are the same 0.5, which is why no factor appears below.
*
* Price impact slopes, $/bbl per kbbl/day of lifting:
*   AL  Arab Light   0.012     widely traded, deep market, least impact
*   BN  Bonny Light  0.020     the light sweet grade the sulphur spec wants most
*   MU  Murban       0.016
*
* Q = diag(-0.012, -0.020, -0.016) is NEGATIVE DEFINITE, so under OBJSENSE MAXIMIZE the
* objective is strictly concave and the problem is a convex QP with a unique optimum.
* src/qp/convexity.cpp tests sense * Q and would REFUSE this file outright if a sign were
* wrong, rather than returning whatever local point an iteration wandered to.
*
* Everything else is inherited unchanged from crude_blend.mps:
*   THRUPUT  CDU throughput, 90 to 120 kbbl/day        (G row widened by RANGES)
*   DIESEL   diesel pool must land exactly on 40       (E row)
*   SULPHUR  0.80 AL - 0.86 BN - 0.22 MU <= 0          (blended sulphur at most 1.00 %wt)
*
* All numeric parameters are invented for this repository - a small, hand-checkable
* instance in the shape of the cited formulation. None of it is real MRPL data.
NAME          CRUDEBLENDQP
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
QUADOBJ
    AL        AL          -0.012
    BN        BN          -0.020
    MU        MU          -0.016
ENDATA
