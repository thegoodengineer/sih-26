* SANKHYA case study - production planning by lot sizing.
*
* PS26119 asks for robustness on 'weak LP relaxations'. This is the textbook source of
* one. Producing in a period costs a fixed set-up charge no matter how little is made,
* and the only way to write that in a MILP is the big-M link
*
*       x_t <= M * y_t,     M = total demand over the horizon
*
* In the RELAXATION y_t is free to take the value x_t / M, so a period producing one
* unit pays 1/180 of a set-up instead of a whole one. The relaxation therefore buys the
* set-up structure at a fraction of its true price and its bound sits far below the
* integer optimum. Branch and bound has to close that gap by search.
*
*   min  sum_t ( 2 x_t + 150 y_t + 1 s_t )
*   s.t. s_{t-1} + x_t - s_t = d_t     (BAL, inventory balance, s_0 = 0)
*        x_t - M y_t <= 0              (LINK)
*        y_t binary
*
*   demand [40.0, 60.0, 30.0, 50.0], M = 180
*
* The demo reports the relaxation bound and the integer optimum side by side, so the
* size of the gap the search actually had to close is visible rather than asserted.
NAME          LOTSIZE
ROWS
 N  TCOST
 E  BAL1
 L  LINK1
 E  BAL2
 L  LINK2
 E  BAL3
 L  LINK3
 E  BAL4
 L  LINK4
COLUMNS
    X1        TCOST                  2  BAL1                   1
    X1        LINK1                  1
    X2        TCOST                  2  BAL2                   1
    X2        LINK2                  1
    X3        TCOST                  2  BAL3                   1
    X3        LINK3                  1
    X4        TCOST                  2  BAL4                   1
    X4        LINK4                  1
    S1        TCOST                  1  BAL1                  -1
    S1        BAL2                   1
    S2        TCOST                  1  BAL2                  -1
    S2        BAL3                   1
    S3        TCOST                  1  BAL3                  -1
    S3        BAL4                   1
    MARKER0000  'MARKER'                 'INTORG'
    Y1        TCOST                150  LINK1               -180
    Y2        TCOST                150  LINK2               -180
    Y3        TCOST                150  LINK3               -180
    Y4        TCOST                150  LINK4               -180
    MARKER0001  'MARKER'                 'INTEND'
RHS
    RHS       BAL1                  40  BAL2                  60
    RHS       BAL3                  30  BAL4                  50
BOUNDS
 BV BND       Y1
 BV BND       Y2
 BV BND       Y3
 BV BND       Y4
ENDATA
