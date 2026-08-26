pub mod cpu;
// CUDA backend intentionally reserved for Milestone 4.
// Future: consume typed IR and emit PTX/cubins.
// The deployment contract must not require the CUDA Toolkit or Python.
pub mod cuda;
