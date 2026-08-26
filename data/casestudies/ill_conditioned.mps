* SANKHYA case study - the supply chain model, deliberately badly scaled.
*
* PS26119 asks for robustness on 'ill-conditioned constraint matrices'. The honest way
* to demonstrate that is on an instance WHOSE ANSWER IS ALREADY KNOWN, so the claim is
* falsifiable rather than a number nobody can check.
*
* This is data/casestudies/supply_chain.mps with row i multiplied by r_i and column j
* substituted x_j -> t_j * x'_j, the objective coefficient scaled to match:
*
*   row scales     10^-6 ... 10^+6
*   column scales  10^-6 ... 10^+4
*
* Both operations are exact re-parameterisations. The feasible set is the same set seen
* through a diagonal change of variables, so
*
*   THE OPTIMAL OBJECTIVE IS UNCHANGED, and    x'_j = x_j / t_j.
*
* Row and column scales COMPOUND, so the entries now span twenty-two orders of
* magnitude - 1e-10 to 1e+12 - and the 2-norm condition number is about 2e28. A solver
* that pivots on
* magnitude alone selects a numerically worthless pivot here, and a solver with no
* scaling declares it infeasible or unbounded. The demo solves both files and checks
* the two objectives agree - if they do not, we have a bug, and it shows.
NAME          ILLCOND
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
    X11       FREIGHT            40000  SUP1                0.01
    X11       DEM1                 0.1
    X12       FREIGHT           0.0006  SUP1               1e-10
    X12       DEM2                   1
    X13       FREIGHT             9000  SUP1               0.001
    X13       DEM3                   1
    X14       FREIGHT            0.005  SUP1               1e-09
    X14       DEM4                 100
    X21       FREIGHT           500000  SUP2               1e+08
    X21       DEM1                   1
    X22       FREIGHT            3e-05  SUP2                0.01
    X22       DEM2                 0.1
    X23       FREIGHT              700  SUP2              100000
    X23       DEM3                 0.1
    X24       FREIGHT             0.08  SUP2                  10
    X24       DEM4                1000
    X31       FREIGHT          6000000  SUP3               1e+12
    X31       DEM1                  10
    X32       FREIGHT            7e-06  SUP3                   1
    X32       DEM2                0.01
    X33       FREIGHT               40  SUP3            10000000
    X33       DEM3                0.01
    X34       FREIGHT              0.3  SUP3              100000
    X34       DEM4               10000
RHS
    RHS       SUP1              0.0003  SUP2              400000
    RHS       SUP3               5e+08  DEM1              0.0025
    RHS       DEM2             3500000  DEM3                 0.4
    RHS       DEM4            20000000
ENDATA
