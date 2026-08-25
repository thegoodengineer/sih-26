\ SANKHYA demo instance - crude blending for a diesel pool.
\
\ The same model as demo/crude_blend.mps, written algebraically. Solving both and getting
\ the same objective, the same recipe and the same shadow prices is the cheapest available
\ check that the two readers agree, and it is what makes the LP reader trustworthy enough
\ to author the Phase 10 case studies in.
\
\ AL Arab Light, BN Bonny Light, MU Murban, in kbbl/day charged to the CDU.

Maximize
 margin: 2.40 AL + 1.60 BN + 1.64 MU

Subject To
 \ CDU throughput window, 90 to 120 kbbl/day
 THRUPUT: 90 <= AL + BN + MU <= 120
 \ the diesel pool must land exactly on its commitment
 DIESEL:  0.30 AL + 0.45 BN + 0.38 MU = 40
 \ blended crude sulphur at most 1.00 %wt, rearranged so the right-hand side is constant
 SULPHUR: 0.80 AL - 0.86 BN - 0.22 MU <= 0

Bounds
 AL >= 10
 BN <= 45
 MU <= 60

End
