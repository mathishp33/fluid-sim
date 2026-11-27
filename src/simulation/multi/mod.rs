//! Solveur MULTIPHASIQUE (docs/MULTIPHASE.md) : plusieurs fluides à interfaces nettes, changements d'état, chimie et flammes, phases
//! granulaires (docs/GRANULAIRE.md), corps rigides (docs/SOLID.md), protocoles d'expérience (docs/EXPERIENCES.md) ; CPU f64
//! (`multiphase::MultiphaseSim`, référence) et GPU f32 (`gpu_multiphase`, `mp_gpu/*.wgsl`).
pub mod mp_advect;
pub mod mp_force;
pub mod mp_pressure;
pub mod mp_step;
pub mod mp_thermo;
pub mod mp_species;
pub mod mp_mg;
pub mod mp_groups;
pub mod mp_granular;
pub mod mp_friction;
pub mod mp_percolation;
pub mod mp_par;
#[cfg(test)]
mod mp_chem_tests;
#[cfg(test)]
mod mp_gpu_tests;
#[cfg(test)]
mod mp_perf_tests;
pub mod mp_cells;
pub mod multiphase;
pub mod phases;
pub mod vof;

#[cfg(test)]
pub mod mp_tests;
pub mod mp_view;
pub mod mp_state;
pub mod mp_bodies;
pub mod mp_mechanism;
pub mod mp_contacts;
pub mod mp_body_heat;
pub mod mp_experiment;
pub mod mp_script;
pub mod mp_fields;
#[cfg(test)]
mod mp_experiment_tests;
pub mod mp_materials;
pub mod gpu_multiphase;
