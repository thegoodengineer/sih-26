# When the GPU arrives: the plan

The team has asked for GPU-accelerated hardware. This is what to do with it the day it comes,
in what order, with what tests, and what it will and will not let us claim. It is written
against `main` at `54fcb6f` (2026-09-16) and the four GPU issues that already define the
track: [#16](https://github.com/thegoodengineer/sih-26/issues/16) plumbing,
[#17](https://github.com/thegoodengineer/sih-26/issues/17) kernels,
[#18](https://github.com/thegoodengineer/sih-26/issues/18) large instances (done),
[#19](https://github.com/thegoodengineer/sih-26/issues/19) evidence. Nothing here replaces
those; it sequences them for a machine we do not have yet and a deadline of 20 September.

The rules that bind every step are the ones in `CLAUDE.md`: no number is reported that was
not produced by a command in the session reporting it; every benchmark writes a CSV to
`bench/results/`; `docs/BENCHMARKS.md` is generated from those CSVs; the CPU build works with
zero CUDA installed; no solver source is read or copied - the GPU design references are the
papers named in #17, not cuPDLP's code.

## 1. What exists today

| piece | state |
|---|---|
| The engine to accelerate | Restarted PDHG on the CPU, `src/pdhg/pdhg.cpp` (747 lines): Ruiz + Pock-Chambolle scaling, adaptive step size, primal weight, restarts, a convergence test every 40 iterations, an interior-point polish of its answer by default (#229). 9 of 9 committed Netlib instances to `optimal` at 1e-8 on `main` at `79ec7f7` (`bench/results/pdhg-79ec7f7.csv`). |
| The instances that make a GPU comparison mean something | `bench/runners/generate_large_lp.py` (#18, done): random and staircase families to a million rows with the optimum exact by construction; `generate_refinery_lp.py` (#211): a refinery planning LP at 12, 365 and 8,760 periods (1,068 / 32,485 / 779,640 rows). Measured on the CPU in `bench/results/scale-*.csv`. |
| The CUDA backend | PR [#153](https://github.com/thegoodengineer/sih-26/pull/153) (Ayush): `src/gpu/pdhg_cuda.cu` (1,143 lines), a device probe, CSR and CSC on the device, fused primal/dual/interaction kernels, batches of 40 iterations between host syncs, a CPU fallback below 20,000 post-presolve nonzeros, a compile-only CUDA CI job. It has never been built, run or measured inside this repository's evidence chain, its description carries no terminal output, and it is 75 commits behind `main` with a 555-line change to the CPU engine's file. |
| The build switch | `SANKHYA_ENABLE_CUDA` in `CMakeLists.txt`, default OFF; `--gpu` warns and runs on the CPU. |
| What the documents say | README, `docs/PS26119_COVERAGE.md` and the demo's section 6 all say GPU acceleration is not started and no speed-up is claimed. That stays true until section 5 below produces a CSV. |

## 2. Day 0: intake, before any code

Two hours, and the output is a file, `docs/GPU_MACHINE.md`, that names the machine the GPU
numbers will belong to. Every later CSV row carries a machine tag, and this is what the tag
points at.

1. **Record the hardware and the toolchain.** `nvidia-smi` (driver version, card, VRAM),
   `nvcc --version`, the compute capability (Ada is 8.9; a different card needs a different
   `CMAKE_CUDA_ARCHITECTURES`), the host compiler `nvcc` will use, the OS. Paste the real
   output into the file.
2. **Know the host-compiler trap before it costs a day.** On Windows, `nvcc` accepts MSVC as
   its host compiler and not MinGW; this repository builds with MSYS2 GCC on the dev boxes.
   Either the GPU machine runs Linux (simplest: `g++` is a supported host), or the CUDA build
   is configured with Visual Studio Build Tools as the host compiler while the CPU build
   keeps GCC. Decide on Day 0, write it down, and do not let the two toolchains share a build
   directory.
3. **Prove the CPU build still works there with zero CUDA in play.** `scripts/configure.sh
   build Release && cmake --build build -j`, then `ctest`. This is the machine's baseline and
   the rule #16 calls non-negotiable.
4. **Run the CPU evidence once on that machine**, alone, on mains: `bench/runners/netlib.py`
   (small tier), `bench/runners/pdhg_report.py`, and `scale.py --engines pdhg` on the random
   family. These CSVs are the CPU side of every GPU comparison; without them a GPU number has
   nothing honest to be compared against, because the laptop the existing CSVs came from is a
   different machine.
5. **Windows only:** confirm the freshly linked binaries run. Smart App Control has blocked
   new executables on the dev box repeatedly (`scripts/preflight.sh` explains the escape
   hatch); on a machine that blocks them, a Debug build or a signed build is needed before
   anything else.

## 3. The order of work

Each step is one PR, merged before the next starts, per the repository's workflow. The
owners are the ones on the issues; the reviewer runs everything listed under "gate" before
merging - the reviews this month found that a description's claims and a branch's behaviour
are different things.

### Step 1 - #16, the plumbing, carved out of #153

Bring only the build and probe parts of #153 to `main` first: the CMake guard with the real
architecture, `src/gpu/device.{hpp,cu}`, `sankhya version` reporting whether CUDA is compiled
in and whether a device is visible, `--gpu` naming the device or logging once and running on
the CPU, the compile-only CI job, and cuSPARSE/cuBLAS in `docs/PROVENANCE.md` with the line
that they are matrix arithmetic, not a solver.

Gate: the four "done when" boxes of #16, with terminal output pasted; the seven existing
CPU-only CI jobs green; `sankhya solve data/netlib/afiro.mps --option algorithm=pdhg --option
gpu=true` returning `-4.6475314286e+02` on the CPU with the "no kernels" path.

### Step 2 - #17, the kernels, on a device

Rebase the rest of #153 onto that. Review it on the machine, in this order:

1. **The CPU engine must be unchanged in what it computes.** #153 touches `src/pdhg/pdhg.cpp`
   heavily. Run `bench/runners/pdhg_report.py` on the branch with the GPU off and diff it
   against the Day-0 CPU CSV: identical iteration counts and objectives on every instance,
   or the diff is explained line by line. A refactor that moves CPU numbers is not a GPU
   backend, it is a different engine.
2. **Agreement, not resemblance.** The test #17 asks for: the same instance solved both ways,
   objectives within 1e-9 relative, skipped cleanly with no device. Run it on all nine
   committed Netlib instances and on the 10,000-row random instance.
3. **The verifier passes on GPU output.** `tools/verify_solution.py` on every `.sol` the GPU
   path writes, including the polished ones: the interior-point polish runs on the CPU after
   the GPU iterations and must still see a point it can finish.
4. **The fallback threshold is a measured number or it goes.** #153 falls back to the CPU
   below 20,000 nonzeros. That is exactly the crossover #19 is supposed to measure; a number
   chosen before the measurement is a guess. Keep the switch, make its default come from
   section 5's table, and say in the option's help where the number came from.
5. **Batching of 40 iterations between syncs must not change the mathematics.** The restart
   and the convergence test read the iterate every 40 iterations on the CPU too; check that
   the batch boundary is that boundary, so the GPU run takes the same decisions on the same
   iterates, up to floating-point reduction order.
6. **Determinism, stated.** GPU reductions are not associative; the PR says what is
   asserted (agreement to a tolerance) and what is not (bit identity), as #17 asks.
7. **Out of memory is a status, not a crash.** #246 found the interior point dying of
   `std::bad_alloc` with no stats file; a `cudaMalloc` failure on a model that does not fit
   6 GB must come back as `numerical_error` with a message naming the size, and the solve
   must not leave the process. Test it with the million-row instance if it does not fit.

Gate: #17's five correctness boxes; every CPU-only CI job green; the CUDA compile job green;
the agreement test's terminal output in the PR.

### Step 3 - #19, the evidence

`bench/runners/gpu_report.py`, next to `pdhg_report.py`, writing one CSV with every column
`CLAUDE.md` requires plus `device`, `vram_bytes`, `iterations`, `gpu_seconds`, `cpu_seconds`
and the tolerance the row was run at. Then a `gpu_section()` in `make_benchmarks_doc.py` so
`docs/BENCHMARKS.md` regenerates from it. The protocol:

- **Alone on the machine, on mains, one run at a time**, the CPU and GPU rows for an instance
  taken back to back. One warm-up solve is run and discarded so that the first CUDA context
  creation is not charged to an instance.
- **Instances, with rows, columns and nonzeros stated on every row:** the nine Netlib
  instances (where the GPU will lose and the table must say so), the random family at 1,000,
  5,000, 20,000, 100,000 and 1,000,000 rows, the staircase family at the same sizes, and the
  refinery year at 12, 365 and 8,760 periods. The ones that do not fit in VRAM are rows too,
  with the status that says why.
- **Two tolerances, never blended:** every instance at 1e-4 and at 1e-8, as
  `pdhg_report.py` already does. A first-order method's cost depends on the accuracy asked.
- **The crossover marked.** The size below which the GPU is slower and above which it is
  faster is the result. If the GPU never wins, that is the result, published.
- **The honest 1e-8 list**, kept current: the instances PDHG cannot drive to 1e-8 in its
  iteration budget on either device (`share2b` is on it today).
- **The VRAM ceiling stated** as the largest instance that fit and the first that did not.
- **Three repeats of the headline rows** an hour apart before any ratio is quoted - the bar
  #212 set for the HiGHS comparison applies here unchanged.

Gate: the CSV committed, the section generated, every number in it from a command run in
the session that committed it, and the README's GPU row rewritten from "unwritten" to what
the table says - including the losing region.

### Step 4 - the documents and the demo

Once the CSV exists: the README status table's Phase 4 row and gaps table's GPU row,
`docs/PS26119_COVERAGE.md`'s "GPU acceleration" line, the demo's section 6 ledger, and a
demo section that solves the 5,000-row instance on the GPU live and prints the crossover
table's headline rows from the CSV. #153 already carries 81 lines of demo change; they are
rewritten to read from the CSV rather than assert.

## 4. What to test, as one list

| kind | test | where |
|---|---|---|
| Build | CPU-only build and every CI job green with no CUDA present; CUDA build compiles and links on the CI runner | #16 |
| Probe | `sankhya version` names the device; `--gpu` with no device logs once and solves on the CPU; `--gpu` with a device names it | #16 |
| Agreement | same instance, CPU and GPU, objectives within 1e-9 relative, on the nine Netlib instances and the 10,000-row random instance; skipped cleanly without a device | #17, `tests/unit/test_pdhg.cpp` |
| Unchanged CPU path | `pdhg_report.py` on the branch with the GPU off is identical to the Day-0 CPU CSV | Step 2 |
| Verification | `tools/verify_solution.py` passes on every GPU `.sol`, polished and unpolished | #17 |
| Batching | restart and convergence decisions taken on the same iterates as the CPU run | Step 2 |
| Failure modes | out of VRAM is a status with a message; a CUDA error mid-solve is a status, never an abort; Ctrl-C (#223) still interrupts a GPU solve at the next batch boundary | Step 2 |
| Fallback threshold | the CPU/GPU switch default comes from the measured crossover, and its help text says so | Step 2 and 3 |
| Performance | the protocol in Step 3: alone, mains, warm-up discarded, both tolerances, three repeats of headline rows, CSV with machine tag | #19 |
| Scale | the largest instance that fits and the first that does not, as rows | #19 |
| Honesty | the losing region and the 1e-8 list are in the generated table; the README quotes nothing the CSV does not contain | #19, `make_benchmarks_doc.py` |

## 5. What the hardware lets us claim, and what it does not

The claim the problem statement's title invites is "GPU-accelerated". After Step 3 the
sentence we can write is of this shape: *on instances above N rows the first-order engine on
the GPU reaches 1e-4 in a fraction F of the CPU's time, on this machine; below N the GPU is
slower; at 1e-8 the picture is T.* Every letter in that sentence comes from the CSV. Until
then the README's "no speed-up is claimed" stands.

What the hardware does not change: the simplex is sequential and stays on the CPU, so
Netlib-sized models get no faster; MIPLIB proofs (13 of 30 reached, 9 proved) depend on
branch-and-cut (#221) and the parallel tree (#222), not on the GPU; Mittelmann's 0 of 8 is a
factorization-scale problem that PDHG at 1e-4 may reach but cannot certify. Those are the
rows in the README's "What still needs doing" that a GPU leaves where they are.

## 6. Better than the existing solvers - where that is true, and where it is not

The honest comparison is by kind. There are three kinds of existing solver: the open-source
simplex codes (HiGHS, CBC, GLPK), the commercial ones (CPLEX, Gurobi, Xpress), and the
GPU first-order codes (PDLP, cuPDLP). Against each, what we have that they do not, with the
evidence that backs it, and what they have that we do not.

**Where we are ahead, on evidence in the repository today**

1. **Every answer is checked by something that shares no code with the solver.**
   `tools/verify_solution.py` reads the `.sol` file and the model and re-derives feasibility,
   duality, integrality and the gap. An `infeasible` verdict carries a Farkas certificate the
   checker proves; an `unbounded` one carries a ray from a feasible point; an IIS carries the
   subsystem's own certificate and one witness per element, so both defining properties are
   checked by arithmetic (#239). No open-source solver ships an independent checker of its
   own output; the commercial ones ship none at all.
2. **The solution file is what a planner reads.** Shadow prices, sensitivity ranges in the
   model's own sense with the basis's degeneracy stated (#249), the smallest conflicting set
   of constraints named when there is no plan (#239), a progress callback bounded at ten a
   second (#234). CPLEX and Gurobi have all of this; the open-source codes have parts of it.
3. **Scale evidence whose optimum is exact by construction.** The random, staircase and
   refinery families (#18, #198, #211) are built backwards from a KKT point, so a
   million-row run is an accuracy check, not a timing loop. Public benchmarks have no such
   families; ours are committed generators with seeds.
4. **Provenance that is enforced, not asserted.** `docs/PROVENANCE.md`, a CI job that fails
   the build if a solver library appears in the binary, and a link-line dump. "Sovereign" is
   a checked property here and a word elsewhere.
5. **An honest ledger as part of the product.** The README's gaps table, the demo's section
   6, the coverage tracker and the "What still needs doing" section say what is missing with
   the number. This is unusual, and a judge who has met vendor benchmarks knows it.

**Where we are behind, on the same evidence**

1. Raw simplex speed: about twice HiGHS's time on the Netlib medium tier
   (`bench/results/compare-highs-medium-f7ca7e9*.csv`), and HiGHS's devex takes fewer
   iterations.
2. MILP: 9 of 30 MIPLIB proofs; no MIR cuts, no cuts below the root, one core.
3. Large LPs: Mittelmann 0 of 8; the interior point runs out of memory on the 100,000-row
   random model (#246); the dual simplex reaches the optimum only at 1,000 rows on the scale
   families (#210, #243).
4. Against PDLP and cuPDLP specifically: they exist, they are published, and they are faster
   than a first port will be. Our GPU engine's claim is not "faster than cuPDLP"; it is "a
   from-scratch GPU first-order engine, verified, with its crossover measured and published
   including where it loses". That is defensible in Q&A; a fabricated 10x is not.

**Where the GPU changes the comparison, if Step 3's table comes out as the design predicts:**
the open-source simplex codes have no GPU path at all, so on models above the crossover
size, at 1e-4, a GPU PDHG run is a capability they do not have. That is the one "better
than" the hardware can add, and it is worth exactly what the table shows.

## 7. The novelty, for the pitch

Say these, in this order, each with the file that proves it:

1. Written from the mathematics, cited above every algorithm (`docs/PROVENANCE.md`), with a
   CI job that would fail the build if that stopped being true.
2. Verification as architecture: an exact rational oracle in the tests, an independent
   checker on every written answer, certificates for the two verdicts that have no point,
   and an IIS that proves both of its own properties.
3. A solver that tells the planner what to do next: which constraints fight, which prices
   the plan survives, which capacity is worth expanding - in the solution file, checked.
4. Scale evidence with exact optima, to a million rows, on three families including a
   refinery year, with the engine-by-engine ceiling stated rather than hidden.
5. Four engines behind one seam (simplex, dual simplex, interior point, PDHG) plus branch and
   bound over LP and QP relaxations, chosen by the problem class, with the first-order engine
   finished by the interior point.
6. After Step 3: a GPU first-order engine whose crossover against its own CPU path is
   measured and published, losing region included.

Do not say: "GPU-accelerated" before the CSV exists; "faster than CPLEX/Gurobi/HiGHS" on
anything but the specific rows a CSV shows; "solves NLP" (the seam exists, the engine does
not, #226).

## 8. Timeline against 20 September

If the hardware arrives with four or more working days left: Day 0 intake; Day 1 Step 1
merged; Day 2 Step 2 reviewed on the device and merged; Day 3 Step 3's runs and CSV, alone
on the machine; Day 4 documents, the demo section, the final read (#213). If it arrives with
less: do Day 0, Step 1 and the agreement test only, and the submission says "the CUDA backend
compiles, links, and agrees with the CPU to 1e-9 on nine instances; its speed is not yet
measured" - which is true, checkable, and better than a number nobody can defend. If it does
not arrive before the deadline, the plan is unchanged and runs after it; the README's GPU row
already says so.

## 9. Risks, named

- **Toolchain on Windows** (section 2, item 2): the most likely way to lose a day.
- **A card other than an RTX 4050**: change the architecture in CMake and the VRAM ceiling
  in the plan; nothing else assumes the card.
- **#153's CPU-path changes**: the Step 2 gate exists because of them.
- **Reduction order**: agreement to a tolerance is the assertion; do not chase bit identity.
- **The clock**: a measured crossover on one machine beats a claimed speed-up on none.
