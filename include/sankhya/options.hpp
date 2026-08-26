// SPDX-License-Identifier: Apache-2.0
// SANKHYA - the option table.
//
// One table drives three surfaces: the CLI (--option name=value), the C API
// (sankhya_set_option_str) and the Python bindings. Adding a knob means adding one row to
// the registry in src/util/options.cpp and nothing else - no parser edit, no CLI edit, no
// binding edit. That is deliberate: every future engine adds options, and three people are
// going to be adding them concurrently.
//
// String-keyed options with typed storage is the surface shape every industrial solver
// exposes. Per CLAUDE.md this is interface compatibility, not derivation.
#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace sankhya {

/// The four value kinds an option can hold.
enum class OptionType { Bool, Int, Double, String };

/// Storage for one option value.
using OptionValue = std::variant<bool, std::int64_t, double, std::string>;

/// Static description of one option: its name, type, default and admissible range.
struct OptionSpec {
  std::string name;
  OptionType type = OptionType::String;
  OptionValue default_value;
  std::string description;
  /// Inclusive numeric bounds. Ignored for Bool and String.
  double min_value = 0.0;
  double max_value = 0.0;
  /// For String options, the permitted values. Empty means unrestricted.
  std::vector<std::string> choices;

  /// Empty when the option is live. Otherwise the phase that will implement it, e.g.
  /// "Phase 6".
  ///
  /// An option table is a capability list, and `sankhya options` is one of the first things
  /// anyone runs. Registering a knob the solver never reads and printing it exactly like a
  /// working one overstates the product - and it is the cheapest kind of overclaim to catch:
  /// set it, watch nothing happen, ask why. The entry stays registered, because the CLI, the
  /// C API and the bindings all read this one table and removing rows would churn all three;
  /// what changes is that the printed table says so.
  std::string planned_for{};

  [[nodiscard]] bool implemented() const { return planned_for.empty(); }
};

/// A mutable set of solver options, initialised from the registry defaults.
class Options {
 public:
  Options();

  /// The full static registry, in declaration order. Used by --help and by the C API.
  static const std::vector<OptionSpec>& registry();

  /// Look up a spec by name; nullptr when unknown.
  static const OptionSpec* find_spec(const std::string& name);

  /// Parse and assign from text, as the CLI and C API do. Returns false and fills `error`
  /// on an unknown name, a malformed value, or a value outside the declared range.
  /// Never throws and never partially assigns.
  bool set_from_string(const std::string& name, const std::string& text, std::string* error);

  /// Typed setters. These assert on a name/type mismatch: a caller inside the solver core
  /// knows the type statically, so a mismatch is a bug rather than user input.
  void set_bool(const std::string& name, bool value);
  void set_int(const std::string& name, std::int64_t value);
  void set_double(const std::string& name, double value);
  void set_string(const std::string& name, const std::string& value);

  /// Typed getters. Assert on unknown names for the same reason.
  [[nodiscard]] bool get_bool(const std::string& name) const;
  [[nodiscard]] std::int64_t get_int(const std::string& name) const;
  [[nodiscard]] double get_double(const std::string& name) const;
  [[nodiscard]] const std::string& get_string(const std::string& name) const;

  /// True when the option exists in the registry.
  [[nodiscard]] static bool exists(const std::string& name);

  /// True when this instance holds a value different from the registry default. Used by
  /// the log banner, which prints only the options the user actually changed.
  [[nodiscard]] bool is_modified(const std::string& name) const;

  /// Names of all options whose value differs from the default, in registry order.
  [[nodiscard]] std::vector<std::string> modified_names() const;

  /// Render the current value as text, in the same syntax set_from_string() accepts.
  [[nodiscard]] std::string value_as_string(const std::string& name) const;

 private:
  const OptionValue& value_of(const std::string& name) const;
  OptionValue& mutable_value_of(const std::string& name);

  /// Values are stored positionally, parallel to registry(). A flat vector beats a map
  /// here: the registry is small, fixed at compile time, and looked up by index once the
  /// spec is resolved.
  std::vector<OptionValue> values_;
};

}  // namespace sankhya
