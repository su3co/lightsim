//! Four-level two-electron gain media on the GPU (`lightsim_core::gain2`):
//! device lists and state, launches of `gain2_pre` / `gain2_post`
//! (`kernels/src/gain2.rs`). Only plans with such media build this.

use super::*;
use anyhow::Context as _;

struct CompDev {
    comp: Component,
    n: usize,
    idx: DeviceBuffer<u64>,
    row: DeviceBuffer<u32>,
    scale: DeviceBuffer<f32>,
    vol: Vec<f64>,
    st: DeviceBuffer<f32>,
    pop: DeviceBuffer<f64>,
}

pub(crate) struct Gain2Dev {
    pre: Function,
    post: Function,
    tab: DeviceBuffer<f64>,
    comps: Vec<CompDev>,
}

pub(crate) fn build(plan: &Plan, module: &Arc<cuda_core::CudaModule>, s: &Arc<CudaStream>) -> Result<Option<Gain2Dev>> {
    let Some(g) = &plan.gain2 else { return Ok(None) };
    let f = |n: &str| module.load_function(n).with_context(|| format!("kernel {n}"));
    let mut comps = vec![];
    for c in &g.comps {
        let n = c.idx.len();
        let mut pop = vec![0f64; 4 * n];
        for (t, &r) in c.row.iter().enumerate() {
            for l in 0..4 {
                pop[l * n + t] = g.init[4 * r as usize + l];
            }
        }
        comps.push(CompDev { comp: c.comp, n, idx: dbuf(s, &c.idx)?, row: dbuf(s, &c.row)?, scale: dbuf(s, &c.scale)?, vol: c.vol.clone(), st: zbuf(s, 7 * n)?, pop: dbuf(s, &pop)? });
    }
    Ok(Some(Gain2Dev { pre: f("gain2_pre")?, post: f("gain2_post")?, tab: dbuf(s, &g.tab)?, comps }))
}

impl CudaBackend {
    pub(crate) fn gain2_pre_launch(&self) -> Result<()> {
        let Some(gd) = &self.gain2 else { return Ok(()) };
        for c in &gd.comps {
            if self.idle_2d(c.comp) {
                continue;
            }
            let args = Args::new().p(self.field_ptr(c.comp)).p(c.idx.cu_deviceptr()).p(c.row.cu_deviceptr()).p(gd.tab.cu_deviceptr()).p(c.pop.cu_deviceptr()).p(c.st.cu_deviceptr()).u(c.n as u32);
            self.launch_1d(&gd.pre, c.n, args)?;
        }
        Ok(())
    }

    pub(crate) fn gain2_post_launch(&self) -> Result<()> {
        let Some(gd) = &self.gain2 else { return Ok(()) };
        for c in &gd.comps {
            if self.idle_2d(c.comp) {
                continue;
            }
            let args = Args::new()
                .p(self.new_ptr(c.comp))
                .p(c.idx.cu_deviceptr())
                .p(c.row.cu_deviceptr())
                .p(gd.tab.cu_deviceptr())
                .p(c.scale.cu_deviceptr())
                .p(c.st.cu_deviceptr())
                .p(c.pop.cu_deviceptr())
                .u(c.n as u32);
            self.launch_1d(&gd.post, c.n, args)?;
        }
        Ok(())
    }

    /// Electrons in levels 0..3 summed over the cells (density times volume).
    pub(crate) fn gain2_levels(&self) -> Result<Option<[f64; 4]>> {
        let Some(gd) = &self.gain2 else { return Ok(None) };
        let mut out = [0f64; 4];
        for c in &gd.comps {
            let pop = c.pop.to_host_vec(&self.stream)?;
            for (l, o) in out.iter_mut().enumerate() {
                *o += (0..c.n).map(|t| pop[l * c.n + t] * c.vol[t]).sum::<f64>();
            }
        }
        Ok(Some(out))
    }
}
