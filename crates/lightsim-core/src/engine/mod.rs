//! Execution backends.
//!
//! A [`Backend`] owns the fields and auxiliary arrays of one slab and executes
//! the elementary operations of a time step. The time loop (single slab or
//! domain-decomposed) lives in [`crate::run`].

use crate::Real;
pub mod cpu;
pub mod fixed_angle;
pub mod reverse;
mod twod;

use crate::grid::Component;
use crate::plan::Plan;
use anyhow::Result;

/// Raw monitor data downloaded from a backend after a run.
#[derive(Clone, Debug, Default)]
pub struct RawDft {
    /// Per box: (re, im) arrays `[freq][local cell]`.
    pub boxes: Vec<(Vec<Real>, Vec<Real>)>,
}

#[derive(Clone, Debug, Default)]
pub struct RawTime {
    /// Per box: samples `[sample][local cell]` flattened.
    pub boxes: Vec<Vec<Real>>,
    pub steps: Vec<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct RawResults {
    pub dft: Vec<RawDft>,
    pub time: Vec<RawTime>,
}

/// Which x ghost plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Lo,
    Hi,
}

/// Elementary operations of the leapfrog step. Order per step `n`:
///
/// 1. `fill_ghost_e` — E ghost (high side) for the H update
/// 2. `update_h` — H^{n-1/2} -> H^{n+1/2} (incl. CPML)
/// 3. `inject_h(n)` — magnetic sources / TFSF H corrections
/// 4. `fill_ghost_h` — H ghost (low side) for the E update
/// 5. `update_e` — E^n -> E^{n+1} (incl. CPML, dispersive poles, chi2/chi3)
/// 6. `finish_e` — off-diagonal tensor coupling (after the dD exchange)
/// 7. `inject_e(n)` — electric sources / TFSF E corrections
/// 8. `record(n)` — DFT and time monitors
pub trait Backend: Send {
    fn name(&self) -> String;
    fn plan(&self) -> &Plan;
    fn fill_ghost_e(&mut self) -> Result<()>;
    fn fill_ghost_h(&mut self) -> Result<()>;
    fn update_h(&mut self) -> Result<()>;
    fn update_e(&mut self) -> Result<()>;
    fn inject_h(&mut self, step: usize) -> Result<()>;
    fn inject_e(&mut self, step: usize) -> Result<()>;
    fn record(&mut self, step: usize) -> Result<()>;
    /// Sum of squared fields over the slab interior (energy proxy).
    fn energy(&mut self) -> Result<f64>;
    /// Block until all queued work is done.
    fn synchronize(&mut self) -> Result<()> {
        Ok(())
    }
    /// True if the backend does its work on the rayon thread pool. The time
    /// loop of such backends runs on a pool thread, so each parallel loop
    /// of a step starts with a cheap fork instead of a hand-over from an
    /// outside thread, which costs more than the step of a small grid.
    fn on_thread_pool(&self) -> bool {
        false
    }
    fn download(&mut self) -> Result<RawResults>;

    // ---- halo exchange (domain decomposition) ----
    /// Read the first (`Side::Lo`) or last (`Side::Hi`) interior x plane of a component.
    fn read_plane(&mut self, comp: Component, side: Side, out: &mut [Real]) -> Result<()>;
    /// Write a ghost x plane of a component.
    fn write_ghost(&mut self, comp: Component, side: Side, data: &[Real]) -> Result<()>;
    /// Download a full field component in storage layout (debug/tests).
    fn field(&mut self, comp: Component) -> Result<Vec<Real>>;

    /// Gain media: number of atoms in levels 0..3 summed over this slab's
    /// gain cells (density times the volume of each E sample); `None` if the
    /// plan has none.
    fn gain_levels(&mut self) -> Result<Option<[f64; 4]>> {
        Ok(None)
    }
    /// Second half of the E update (off-diagonal tensor coupling), after the
    /// chain has exchanged the dD boundary planes.
    fn finish_e(&mut self) -> Result<()> {
        Ok(())
    }
    /// Off-diagonal tensors: read the dD (`which` = 0) or E^n (`which` = 1)
    /// plane of `comp` at the first (`Lo`) or last (`Hi`) interior x plane.
    fn tensor_read(&mut self, _comp: Component, _side: Side, _which: usize, _out: &mut [Real]) -> Result<()> {
        Ok(())
    }
    /// Off-diagonal tensors: write a dD / E^n ghost plane (`Lo`: 0, `Hi`: nxl+1).
    fn tensor_write_ghost(&mut self, _comp: Component, _side: Side, _which: usize, _data: &[Real]) -> Result<()> {
        Ok(())
    }

    /// Time reversal, E^{n+1} -> E^n (needs H^{n+1/2}): the leapfrog run
    /// backwards with inverted coefficients. See [`crate::reverse`].
    fn reverse_e(&mut self) -> Result<()> {
        anyhow::bail!("time reversal is not supported by this backend")
    }
    /// Time reversal, H^{n+1/2} -> H^{n-1/2} (needs E^n).
    fn reverse_h(&mut self) -> Result<()> {
        anyhow::bail!("time reversal is not supported by this backend")
    }
    /// Overwrite a full field component (storage layout, current set).
    fn set_field(&mut self, _comp: Component, _data: &[Real]) -> Result<()> {
        anyhow::bail!("set_field is not supported by this backend")
    }

    /// Bloch (complex-field) runs: make the imaginary (`true`) or the real
    /// field set current for `read_plane`/`write_ghost`/`device_plane`.
    fn select_set(&mut self, _imag: bool) -> Result<()> {
        Ok(())
    }

    // ---- optional device-to-device halo exchange ----
    /// Device address and context handle of a storage x plane (0..=nxl+1) of a
    /// component, for peer-to-peer copies between GPUs of one process.
    fn device_plane(&self, _comp: Component, _storage_plane: usize) -> Option<(u64, usize)> {
        None
    }
    /// Copy one plane from a peer device address into storage plane
    /// `dst_plane` of `comp` (stream-ordered on this backend).
    fn copy_plane_from_device(&mut self, _comp: Component, _dst_plane: usize, _src: (u64, usize)) -> Result<()> {
        anyhow::bail!("device-to-device copy not supported by this backend")
    }
}
