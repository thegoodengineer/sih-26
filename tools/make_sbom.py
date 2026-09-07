#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate docs/sbom.spdx.json, the software bill of materials, from CMakeLists.txt (#73).

WHY THIS IS GENERATED RATHER THAN WRITTEN. PS26119 asks for a sovereign, auditable foundation,
and the audit a reader can actually perform is "what is in this binary, and is any of it a
solver". That question is answered by the build file, not by prose: every third-party
dependency arrives through a `FetchContent_Declare` block with a pinned tag. So the SBOM is
read out of those blocks. A dependency added to the build without a corresponding entry here
makes `--check` fail, the same way docs/BENCHMARKS.md cannot drift from the CSVs it is
generated from.

The licence and the one-line description of each dependency come from the table in
docs/PROVENANCE.md, which is the file the red-line policy is kept in; they are matched by
name, and a dependency the build declares but that table does not mention is an error rather
than an omission.

    python tools/make_sbom.py            # write docs/sbom.spdx.json
    python tools/make_sbom.py --check    # exit 1 if the committed file is out of date

`--check` compares everything except `creationInfo.created`: a wall-clock stamp would make
the file differ from itself on every regeneration, which is not drift.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CMAKE = REPO_ROOT / "CMakeLists.txt"
PROVENANCE = REPO_ROOT / "docs" / "PROVENANCE.md"
OUTPUT = REPO_ROOT / "docs" / "sbom.spdx.json"

# The name in the build file is not always the name a human uses. Left: the FetchContent
# target. Right: (display name, the row it is documented under in PROVENANCE's table).
DISPLAY = {
    "fmt": ("fmt", "fmt"),
    "cli11": ("CLI11", "CLI11"),
    "nlohmann_json": ("nlohmann/json", "nlohmann/json"),
    "zlib": ("zlib", "zlib"),
    "googletest": ("GoogleTest", "GoogleTest"),
}

# Dependencies that are not fetched by the build and so cannot be read out of it: the
# compiler's own OpenMP runtime, found by `find_package(OpenMP)` when it is present.
RUNTIME_ONLY = [
    {
        "name": "OpenMP runtime",
        "version": "as shipped by the compiler",
        "license": "GPL-3.0 WITH GCC-exception-3.1 OR Apache-2.0 WITH LLVM-exception",
        "download": "NOASSERTION",
        "purl": None,
        "note": "optional, linked only when SANKHYA_WITH_OPENMP finds it",
    },
]


def project_version() -> str:
    text = CMAKE.read_text(encoding="utf-8")
    match = re.search(r"VERSION\s+(\d+\.\d+\.\d+)", text)
    return match.group(1) if match else "0.0.0"


def declared_dependencies() -> list[dict]:
    """Every FetchContent_Declare block, in the order the build file gives them."""
    text = CMAKE.read_text(encoding="utf-8")
    out = []
    for block in re.finditer(
        r"FetchContent_Declare\(\s*(\w+)\s+GIT_REPOSITORY\s+(\S+)\s+GIT_TAG\s+(\S+)", text
    ):
        target, repository, tag = block.group(1), block.group(2), block.group(3)
        display, provenance_name = DISPLAY.get(target, (target, target))
        out.append({
            "target": target,
            "name": display,
            "provenance_name": provenance_name,
            "version": tag.lstrip("v"),
            "download": repository,
            "tag": tag,
        })
    return out


def provenance_rows() -> dict[str, dict]:
    """The dependency table of docs/PROVENANCE.md, keyed by the bolded name in column one."""
    rows: dict[str, dict] = {}
    for line in PROVENANCE.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        name = cells[0].strip("*` ")
        name = name.split(" (")[0]
        rows[name] = {"version": cells[1], "license": cells[2], "linked": cells[3],
                      "why": cells[4]}
    return rows


def spdx_id(name: str) -> str:
    return "SPDXRef-Package-" + re.sub(r"[^A-Za-z0-9.-]", "-", name)


def build_document() -> dict:
    version = project_version()
    deps = declared_dependencies()
    table = provenance_rows()

    missing = [d["provenance_name"] for d in deps if d["provenance_name"] not in table]
    if missing:
        raise SystemExit(
            "these dependencies are declared in CMakeLists.txt but are not in the dependency "
            "table of docs/PROVENANCE.md, which is where the red-line policy is kept: "
            + ", ".join(missing))

    created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    packages = [{
        "name": "sankhya",
        "SPDXID": spdx_id("sankhya"),
        "versionInfo": version,
        "downloadLocation": "https://github.com/thegoodengineer/sih-26",
        "filesAnalyzed": False,
        "licenseConcluded": "Apache-2.0",
        "licenseDeclared": "Apache-2.0",
        "copyrightText": "NOASSERTION",
        "supplier": "Organization: SANKHYA (SIH 2026 PS26119)",
        "description": "A mathematical optimization solver core (LP, MILP, convex QP, MIQP) "
                       "written from mathematical foundations in C++20.",
    }]
    relationships = [{
        "spdxElementId": "SPDXRef-DOCUMENT",
        "relationshipType": "DESCRIBES",
        "relatedSpdxElement": spdx_id("sankhya"),
    }]

    for dep in deps:
        row = table[dep["provenance_name"]]
        package = {
            "name": dep["name"],
            "SPDXID": spdx_id(dep["name"]),
            "versionInfo": dep["version"],
            "downloadLocation": dep["download"],
            "filesAnalyzed": False,
            "licenseConcluded": row["license"],
            "licenseDeclared": row["license"],
            "copyrightText": "NOASSERTION",
            "supplier": "NOASSERTION",
            "description": row["why"],
            "comment": f"linked: {row['linked']}; pinned in CMakeLists.txt at {dep['tag']}",
        }
        repo_path = dep["download"].removeprefix("https://github.com/").removesuffix(".git")
        if "/" in repo_path:
            owner, name = repo_path.split("/", 1)
            package["externalRefs"] = [{
                "referenceCategory": "PACKAGE-MANAGER",
                "referenceType": "purl",
                "referenceLocator": f"pkg:github/{owner}/{name}@{dep['tag']}",
            }]
        packages.append(package)
        relationships.append({
            "spdxElementId": spdx_id("sankhya"),
            "relationshipType": "DEPENDS_ON",
            "relatedSpdxElement": spdx_id(dep["name"]),
        })

    for dep in RUNTIME_ONLY:
        packages.append({
            "name": dep["name"],
            "SPDXID": spdx_id(dep["name"]),
            "versionInfo": dep["version"],
            "downloadLocation": dep["download"],
            "filesAnalyzed": False,
            "licenseConcluded": dep["license"],
            "licenseDeclared": dep["license"],
            "copyrightText": "NOASSERTION",
            "supplier": "NOASSERTION",
            "comment": dep["note"],
        })
        relationships.append({
            "spdxElementId": spdx_id("sankhya"),
            "relationshipType": "OPTIONAL_DEPENDENCY_OF",
            "relatedSpdxElement": spdx_id(dep["name"]),
        })

    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": f"sankhya-{version}",
        "documentNamespace": f"https://github.com/thegoodengineer/sih-26/spdx/sankhya-{version}",
        "creationInfo": {
            "created": created,
            "creators": ["Tool: tools/make_sbom.py", "Organization: SANKHYA (SIH 2026 PS26119)"],
            "comment": "Generated from the FetchContent_Declare blocks of CMakeLists.txt and "
                       "the dependency table of docs/PROVENANCE.md. Not a binary scan: it "
                       "states what the build declares, which is the thing a reader can check "
                       "against the source. `ldd` output for the built binary is printed by "
                       "the provenance job of .github/workflows/ci.yml and is the "
                       "complementary check.",
        },
        "packages": packages,
        "relationships": relationships,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if docs/sbom.spdx.json is out of date")
    args = parser.parse_args()

    document = build_document()
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not OUTPUT.exists():
            print(f"{OUTPUT.relative_to(REPO_ROOT)} does not exist; run tools/make_sbom.py",
                  file=sys.stderr)
            return 1
        committed = json.loads(OUTPUT.read_text(encoding="utf-8"))
        fresh = json.loads(rendered)
        committed.get("creationInfo", {}).pop("created", None)
        fresh.get("creationInfo", {}).pop("created", None)
        if committed != fresh:
            print("docs/sbom.spdx.json is out of date with CMakeLists.txt / PROVENANCE.md.\n"
                  "Regenerate it:  python tools/make_sbom.py", file=sys.stderr)
            return 1
        print(f"docs/sbom.spdx.json is up to date "
              f"({len(fresh['packages'])} packages, SPDX 2.3)")
        return 0

    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} "
          f"({len(document['packages'])} packages, SPDX 2.3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
