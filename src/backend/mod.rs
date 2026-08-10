pub mod cpu;
// CUDA backend intentionally reserved for Milestone 4.
// Future: lower the typed AST into a device-neutral IR, then emit PTX/cubins.
// The deployment contract must not require the CUDA Toolkit or Python.
pub mod cuda;
