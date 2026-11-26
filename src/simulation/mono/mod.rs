//! Solveur MONOPHASIQUE : un fluide (gaz low-Mach ou liquide), température, solides immobiles ; sur le CPU (`cpu`) ou le GPU (`gpu`,
//! wgpu), les deux pilotés par `fluid::FluidSim`.
pub mod fluid;
pub mod gpu;
pub mod cpu;

#[cfg(test)]
pub mod benchmarks;
#[cfg(test)]
pub mod lowmach_regression_tests;
