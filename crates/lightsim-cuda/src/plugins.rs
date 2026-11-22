//! User-defined material plug-ins on the GPU (`lightsim_core::plugin`).
//! The plug-ins run on the host: each step the cells of every plug-in
//! medium are gathered from the device (E^n before the update, E_lin after
//! it), handed to the plug-in, and its E^{n+1} is scattered back. This costs
//! two device-host round trips per plug-in list and step (measured in
//! `docs/guide/materials.md`); plans without plug-in media build nothing.

use super::*;
use anyhow::Context as _;
use lightsim_core::Real;

struct ListDev {
    comp: Component,
    n: usize,
    idx: DeviceBuffer<u64>,
    buf: DeviceBuffer<f32>,
    run: lightsim_core::plugin::PluginRun,
    name: String,
    medium: String,
    inv_eps: Vec<Real>,
}

pub(crate) struct PluginDev {
    gather: Function,
    scatter: Function,
    lists: Vec<ListDev>,
    step: usize,
}

pub(crate) fn build(plan: &Plan, module: &Arc<cuda_core::CudaModule>, s: &Arc<CudaStream>) -> Result<Option<PluginDev>> {
    let Some(pp) = &plan.plugin else { return Ok(None) };
    let f = |n: &str| module.load_function(n).with_context(|| format!("kernel {n}"));
    let mut lists = vec![];
    for l in &pp.lists {
        lists.push(ListDev {
            comp: l.comp,
            n: l.idx.len(),
            idx: dbuf(s, &l.idx)?,
            buf: zbuf(s, l.idx.len())?,
            run: lightsim_core::plugin::PluginRun::new(l)?,
            name: l.plugin.name.clone(),
            medium: l.medium.clone(),
            inv_eps: l.inv_eps.clone(),
        });
    }
    Ok(Some(PluginDev { gather: f("plugin_gather")?, scatter: f("plugin_scatter")?, lists, step: 0 }))
}

impl CudaBackend {
    /// Gather the list's cells of `field` into `buf` and copy them to `out`.
    fn plugin_download(&self, pd: &PluginDev, l: &ListDev, field: u64, out: &mut [Real]) -> Result<()> {
        self.launch_1d(&pd.gather, l.n, Args::new().p(field).p(l.idx.cu_deviceptr()).p(l.buf.cu_deviceptr()).u(l.n as u32))?;
        unsafe { memcpy_dtoh_async(out.as_mut_ptr(), l.buf.cu_deviceptr(), l.n * 4, self.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Plug-in media before the E update: E^n to the host, `before`.
    pub(crate) fn plugin_pre(&mut self) -> Result<()> {
        let Some(mut pd) = self.plugins.take() else { return Ok(()) };
        let r = (|| -> Result<()> {
            for i in 0..pd.lists.len() {
                if self.idle_2d(pd.lists[i].comp) {
                    continue;
                }
                let field = self.field_ptr(pd.lists[i].comp);
                let mut e_prev = std::mem::take(&mut pd.lists[i].run.e_prev);
                self.plugin_download(&pd, &pd.lists[i], field, &mut e_prev)?;
                let l = &mut pd.lists[i];
                l.run.e_prev = e_prev;
                let call = lightsim_core::plugin::PluginCall { axis: l.comp.axis(), dt: self.plan.dt, step: pd.step };
                l.run.plugin.before(&call, &l.run.e_prev, &mut l.run.state);
                if let Some(e) = l.run.plugin.error() {
                    anyhow::bail!("material plug-in '{}' (medium '{}'): {e}", l.name, l.medium);
                }
            }
            Ok(())
        })();
        self.plugins = Some(pd);
        r
    }

    /// Plug-in media after the E update: E_lin to the host, `after`, and
    /// E^{n+1} back.
    pub(crate) fn plugin_post(&mut self) -> Result<()> {
        let Some(mut pd) = self.plugins.take() else { return Ok(()) };
        let r = (|| -> Result<()> {
            for i in 0..pd.lists.len() {
                if self.idle_2d(pd.lists[i].comp) {
                    continue;
                }
                let field = self.new_ptr(pd.lists[i].comp);
                let mut e = std::mem::take(&mut pd.lists[i].run.e);
                self.plugin_download(&pd, &pd.lists[i], field, &mut e)?;
                let l = &mut pd.lists[i];
                l.run.e = e;
                let call = lightsim_core::plugin::PluginCall { axis: l.comp.axis(), dt: self.plan.dt, step: pd.step };
                l.run.plugin.after(&call, &mut l.run.e, &l.run.e_prev, &l.inv_eps, &mut l.run.state);
                if let Some(e) = l.run.plugin.error() {
                    anyhow::bail!("material plug-in '{}' (medium '{}'): {e}", l.name, l.medium);
                }
                let l = &pd.lists[i];
                unsafe { memcpy_htod_async(l.buf.cu_deviceptr(), l.run.e.as_ptr(), l.n * 4, self.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
                self.launch_1d(&pd.scatter, l.n, Args::new().p(field).p(l.idx.cu_deviceptr()).p(l.buf.cu_deviceptr()).u(l.n as u32))?;
                self.stream.synchronize()?;
            }
            pd.step += 1;
            Ok(())
        })();
        self.plugins = Some(pd);
        r
    }
}
