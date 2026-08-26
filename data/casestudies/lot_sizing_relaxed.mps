* SANKHYA case study - the LP RELAXATION of lot_sizing.mps, y continuous in [0, 1].
*
* Emitted as its own file so the weak-relaxation claim is MEASURED rather than
* asserted. Solving both and printing the two objectives shows the exact size of the
* gap branch and bound had to close by search.
NAME          LOTSIZLP
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
    Y1        TCOST                150  LINK1               -180
    Y2        TCOST                150  LINK2               -180
    Y3        TCOST                150  LINK3               -180
    Y4        TCOST                150  LINK4               -180
RHS
    RHS       BAL1                  40  BAL2                  60
    RHS       BAL3                  30  BAL4                  50
BOUNDS
 UP BND       Y1                     1
 UP BND       Y2                     1
 UP BND       Y3                     1
 UP BND       Y4                     1
ENDATA
