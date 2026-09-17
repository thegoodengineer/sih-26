// SPDX-License-Identifier: Apache-2.0
// SANKHYA - model readers and result writers.
//
// Readers return a status object rather than throwing. A malformed model file is ordinary
// user input, not an exceptional condition, and the CLI, the C API and the Python bindings
// all want the same "where and why" string. Every error carries the file and the 1-based
// line number so a judge pointing us at a broken instance gets a usable message.
//
// Every reader calls Model::validate() before returning success. Per ENGINEERING_RULES.md a
// reader that emits a subtly malformed model produces a plausible-looking wrong optimum rather
// than a crash, and that is the failure mode this project is most exposed to.
#pragma once

#include <string>

#include "sankhya/model.hpp"
#include "sankhya/options.hpp"

namespace sankhya::io {

/// Outcome of a read. `ok == false` means `model` is in an unspecified state and must be
/// discarded.
struct ReadResult {
  bool ok = false;
  std::string error;  ///< "path:line: message", empty when ok

  [[nodiscard]] explicit operator bool() const noexcept { return ok; }

  static ReadResult success() { return ReadResult{true, {}}; }
  static ReadResult failure(std::string message) {
    return ReadResult{false, std::move(message)};
  }
};

/// Which MPS dialect to parse. `kAuto` reads the file with the whitespace tokenizer and
/// falls back to the column-oriented one when that produces a structurally impossible line,
/// which is the only situation in which the distinction is observable: fixed-format files
/// whose names contain embedded blanks.
enum class MpsFormat { kAuto, kFree, kFixed };

[[nodiscard]] bool parse_mps_format(const std::string& text, MpsFormat* out) noexcept;

/// Read an MPS file into `model`, replacing its contents. Handles both dialects, and gzip
/// input when the build has zlib. `format_used` receives kFree or kFixed when non-null.
ReadResult read_mps(const std::string& path, Model* model, MpsFormat format = MpsFormat::kAuto,
                    MpsFormat* format_used = nullptr);

/// Read a CPLEX LP file into `model`, replacing its contents.
ReadResult read_lp(const std::string& path, Model* model);

/// Read a model, choosing the reader from the file extension. `.mps`, `.lp`, and either
/// with a `.gz` suffix. An unrecognised extension is read as MPS.
ReadResult read_model(const std::string& path, Model* model);

// -----------------------------------------------------------------------------------------
// Writers
// -----------------------------------------------------------------------------------------

/// Write the solution in SANKHYA's text solution format. Values are printed with 17
/// significant digits so that a reader recovers the exact double: tools/verify_solution.py
/// recomputes the objective from these numbers and compares against ours at 1e-9, which is
/// only meaningful if the file round-trips bit-for-bit.
///
/// Returns false and fills `error` on an I/O failure.
bool write_solution(const std::string& path, const Model& model, const Solution& solution,
                    std::string* error);

/// The same, recording the MIP gap targets the solve ran with in the header
/// (`mip_relative_gap`, `mip_absolute_gap`), so that a verifier can hold an `optimal` MILP to
/// the tolerance that was actually requested rather than to the project default (#188).
bool write_solution(const std::string& path, const Model& model, const Solution& solution,
                    const Options& options, std::string* error);

/// Write a machine-readable JSON result blob: status, objective, measured infeasibilities,
/// effort counters and build identification. This is what the benchmark runners parse, so
/// its keys are part of the interface and must not be renamed casually.
///
/// `options`, when given, adds the "limits" block: the limits the solve was CONFIGURED with
/// beside the counters it reached, so a record that says node_limit can be read without the
/// command line that produced it (#289).
bool write_stats_json(const std::string& path, const Model& model, const Solution& solution,
                      std::string* error, const Options* options = nullptr);

// -----------------------------------------------------------------------------------------
// Model writers (inverse of the readers above)
// -----------------------------------------------------------------------------------------

/// Write the model in free-format MPS. Handles LPs, MILPs and convex QPs (QUADOBJ section).
/// Round-trips via read_mps: dimensions, nonzeros at full precision, bounds and integrality
/// are all recovered exactly.
///
/// Limitation: neither MPS nor LP can express a free constraint (no finite bound on either
/// side). Free rows are written as extra N rows in MPS, but MPS readers universally drop all
/// N rows except the first (the objective). A warning is printed to stderr with the count.
/// Presolve removes free rows before they reach write_presolved, so this affects only raw
/// models that carry them.
///
/// Returns false and fills `error` on an I/O failure.
bool write_mps(const std::string& path, const Model& model, std::string* error);

/// Write the model in CPLEX LP dialect. Ranged rows use the `lo <= expr <= hi` syntax the
/// reader accepts.
///
/// Limitation: LP files cannot encode a quadratic objective (the reader rejects `[ ... ] / 2`
/// today); returns false with an error message when `model.has_quadratic_objective()`. Use
/// write_mps for QP models. Free rows are skipped (no LP syntax for them); a warning is
/// printed to stderr with the count.
///
/// Returns false and fills `error` on an I/O failure.
bool write_lp(const std::string& path, const Model& model, std::string* error);

/// Write the model choosing the format from the file extension: `.lp` -> LP, everything else
/// -> MPS. This is the function model.write() dispatches to from the C API and Python.
///
/// Returns false and fills `error` on an I/O failure.
bool write_model(const std::string& path, const Model& model, std::string* error);

}  // namespace sankhya::io
