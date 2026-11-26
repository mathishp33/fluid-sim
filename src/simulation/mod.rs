//! Simulation : le solveur monophasique (`mono` : un fluide, CPU ou GPU), le solveur multiphasique (`multi` : plusieurs fluides,
//! changements d'état, chimie, grains, corps rigides ; CPU f64 et GPU f32) et les lois de propriétés communes (`materials`).
pub mod materials;
pub mod mono;
pub mod multi;
