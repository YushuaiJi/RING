# Native runtime provenance

The implementation under `cpp/bounded_blowup/` was written independently for
the PVLDB2027-RFANN Bounded-Blowup method. It does not copy implementation code
from the project's baselines or from an external ANN library.

The extension uses pybind11 only as a Python/C++ binding dependency. pybind11 is
provided by the remote build environment under its own BSD-3-Clause license;
no pybind11 source is vendored here. The core router, validator, budgeted graph
search, mmap loader and tests in this directory are project-original code,
released under the MIT License (see `LICENSE` at the repository root) and
marked with `SPDX-License-Identifier: MIT`.

Compiler, dependency versions, source-tree SHA-256 and produced extension
SHA-256 must be captured by the remote Slurm native-build receipt before the
binary is eligible for experiments.
