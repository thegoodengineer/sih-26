* SANKHYA demo instance - the crude blending model with a mode-switch binary.
*
* Same three crudes as demo/crude_blend.mps, plus one binary: whether the CDU runs in
* max-diesel mode. Running that mode costs 12 units of margin and relaxes the sulphur
* specification, which is the shape a real refinery scheduling model takes.
*
* THIS FILE IS EXPECTED TO BE REFUSED until Phase 5 lands branch and cut. Handing it to the
* LP engine and reporting the fractional relaxation as optimal is the single most damaging
* thing the dispatcher could do, so CI solves this instance on every pull request purely to
* confirm it comes back `not_solved` rather than with a plausible answer.
*
* MODE is the binary. Linked to the sulphur row by a big-M term: when MODE = 1 the sulphur
* constraint is slackened by 20 units.
NAME          BLENDMILP
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
    MARKER                 'MARKER'                 'INTORG'
    MODE      MARGIN     -12.00   SULPHUR    -20.00
    MARKER                 'MARKER'                 'INTEND'
RHS
    RHS       THRUPUT     90.00   DIESEL      40.00
    RHS       SULPHUR      0.00
RANGES
    RNG       THRUPUT     30.00
BOUNDS
 LO BND       AL          10.00
 UP BND       BN          45.00
 UP BND       MU          60.00
 BV BND       MODE
ENDATA
