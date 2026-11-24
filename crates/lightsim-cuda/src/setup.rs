//! Set-up of a [`CudaBackend`]: the kernels, and the plan uploaded into
//! device buffers (fields, coefficients, CPML, sources, dispersive,
//! nonlinear, gain and tensor lists, monitors), and the inverted
//! coefficients of time reversal.

use super::*;

pub(crate) fn hloss_dev(s: &Arc<CudaStream>, hl: &[lightsim_core::plan::HLoss; 3]) -> Result<[HLossDev; 3]> {
    let one = |l: &lightsim_core::plan::HLoss| -> Result<HLossDev> { Ok(HLossDev { idx: dbuf(s, &l.idx)?, da: dbuf(s, &l.da)?, n: l.idx.len() }) };
    Ok([one(&hl[0])?, one(&hl[1])?, one(&hl[2])?])
}

/// Coefficients of the reversed leapfrog: `-cb/ca`, `1/ca`, `-ch` (divided
/// by `da` on magnetically lossy cells, whose `da` becomes `1/da`).
pub(crate) fn reverse_coefs(p: &Plan, s: &Arc<CudaStream>) -> Result<RevCoefs> {
    let cb = [0, 1, 2].map(|a| p.cb[a].iter().enumerate().map(|(i, &b)| -b / p.ca.as_ref().map(|c| c[a][i]).unwrap_or(1.0)).collect::<Vec<f32>>());
    let cb = [dbuf(s, &cb[0])?, dbuf(s, &cb[1])?, dbuf(s, &cb[2])?];
    let ca = match &p.ca {
        Some(c) => {
            let inv = |a: usize| c[a].iter().map(|&v| 1.0 / v).collect::<Vec<f32>>();
            Some([dbuf(s, &inv(0))?, dbuf(s, &inv(1))?, dbuf(s, &inv(2))?])
        }
        None => None,
    };
    let mut chn = p.ch.as_ref().map(|c| [0, 1, 2].map(|a| c[a].iter().map(|&v| -v).collect::<Vec<f32>>()));
    let mut hl = p.hloss.clone();
    if let (Some(c), Some(h)) = (chn.as_mut(), hl.as_mut()) {
        lightsim_core::engine::cpu::reverse_hloss(c, h);
    }
    let ch = match &chn {
        Some(c) => Some([dbuf(s, &c[0])?, dbuf(s, &c[1])?, dbuf(s, &c[2])?]),
        None => None,
    };
    let hloss = match &hl {
        Some(h) => Some(hloss_dev(s, h)?),
        None => None,
    };
    Ok(RevCoefs { cb, ca, ch, hloss })
}

pub(crate) fn dbuf<T: cuda_core::DeviceCopy>(stream: &Arc<CudaStream>, data: &[T]) -> Result<DeviceBuffer<T>> {
    if data.is_empty() {
        return Ok(DeviceBuffer::<T>::zeroed(stream, 1)?);
    }
    Ok(DeviceBuffer::from_host(stream, data)?)
}

pub(crate) fn zbuf(stream: &Arc<CudaStream>, n: usize) -> Result<DeviceBuffer<f32>> {
    Ok(DeviceBuffer::<f32>::zeroed(stream, n.max(1))?)
}

impl CudaBackend {
    /// Create a backend for `plan` on GPU `ordinal`.
    pub fn new(plan: Plan, ordinal: usize) -> Result<CudaBackend> {
        Self::new_with(plan, ordinal, true)
    }

    /// As [`new`](Self::new); `allow_fused = false` forces the split kernels
    /// (needed for time reversal, which runs the split update backwards).
    pub fn new_with(plan: Plan, ordinal: usize, allow_fused: bool) -> Result<CudaBackend> {
        let (ctx, module) = device_module(ordinal)?;
        let device_name = ctx.name().unwrap_or_else(|_| "GPU".into());
        let stream = ctx.new_stream()?;
        let k = load_kernels(&module)?;
        let len = plan.storage_len();
        let s = &stream;
        let e = [zbuf(s, len)?, zbuf(s, len)?, zbuf(s, len)?];
        let h = [zbuf(s, len)?, zbuf(s, len)?, zbuf(s, len)?];
        let cb = [dbuf(s, &plan.cb[0])?, dbuf(s, &plan.cb[1])?, dbuf(s, &plan.cb[2])?];
        let ca = match &plan.ca {
            Some(c) => Some([dbuf(s, &c[0])?, dbuf(s, &c[1])?, dbuf(s, &c[2])?]),
            None => None,
        };
        let ch = match &plan.ch {
            Some(c) => Some([dbuf(s, &c[0])?, dbuf(s, &c[1])?, dbuf(s, &c[2])?]),
            None => None,
        };
        let hloss = match &plan.hloss {
            Some(h) => Some(hloss_dev(s, h)?),
            None => None,
        };
        let inv_p = [dbuf(s, &plan.inv_p[0])?, dbuf(s, &plan.inv_p[1])?, dbuf(s, &plan.inv_p[2])?];
        let inv_d = [dbuf(s, &plan.inv_d[0])?, dbuf(s, &plan.inv_d[1])?, dbuf(s, &plan.inv_d[2])?];
        let pml = pml_dev(s, &plan)?;
        let mk_inj = |v: &Vec<lightsim_core::plan::Injection>| injections_dev(s, v);
        let inj_h = mk_inj(&plan.inj_h)?;
        let inj_e = mk_inj(&plan.inj_e)?;
        let swap_amps = |v: &Vec<lightsim_core::plan::Injection>| swapped_amps(s, v);
        // (complex permittivity tensors run the split kernels)
        let fused_state = if allow_fused && plan.herm.is_none() { fused::build(&plan, &stream, &mk_inj, &swap_amps, ctx.multiprocessor_count().unwrap_or(0))? } else { None };
        let cplx = imag_set(s, &plan)?;
        let split_tab = split_table(s, &plan, fused_state.is_some())?;
        let bsign = [-(plan.bloch_phase[0].1 as f32), -(plan.bloch_phase[1].1 as f32), -(plan.bloch_phase[2].1 as f32)];
        let disp_dense = disp_dense_dev(s, &plan, &fused_state)?;
        let disp = disp_dev(s, &plan, disp_dense.is_some())?;
        let nl = nonlinear_dev(s, &plan)?;
        let raman = raman_dev(s, &plan)?;
        let gain = gain_dev(s, &plan)?;
        let tensor = tensor_dev(s, &plan)?;
        let dft = dft_dev(s, &plan)?;
        let time = time_dev(&plan);
        let sheet = sheets::build(&plan, &module, s)?;
        let plugins = plugins::build(&plan, &module, s)?;
        let gain2 = gain2::build(&plan, &module, s)?;
        let herm = herm::build(&plan, &module, s)?;
        let energy = zbuf(s, plan.plane())?;
        let dummy = zbuf(s, 1)?;
        stream.synchronize()?;
        let nz = plan.nz() as u32;
        let bx = if nz >= 32 { 32 } else { nz.next_power_of_two().max(1) };
        let by = (256 / bx).min((plan.ny() as u32).next_power_of_two()).max(1);
        // 2-D mode: one DFT launch per group of monitors (LIGHTSIM_2D_BATCH=0: one per box)
        let batch = std::env::var("LIGHTSIM_2D_BATCH").as_deref() != Ok("0");
        let dft_groups = if batch && fused_state.as_ref().map(|f| f.twod.is_some()).unwrap_or(false) {
            let ptrs = |m: usize, b: usize, imag: bool| -> (u64, u64) {
                match (&cplx, imag) {
                    (Some(c), true) => (c.dft[m][b].0.cu_deviceptr(), c.dft[m][b].1.cu_deviceptr()),
                    _ => (dft[m].boxes[b].re.cu_deviceptr(), dft[m].boxes[b].im.cu_deviceptr()),
                }
            };
            twod::dft_groups(&plan, s, cplx.is_some(), &ptrs)?
        } else {
            vec![]
        };
        stream.synchronize()?;
        Ok(CudaBackend {
            plan,
            ctx,
            stream,
            _module: module,
            k,
            e,
            h,
            cb,
            ca,
            ch,
            hloss,
            inv_p,
            inv_d,
            pml,
            inj_h,
            inj_e,
            disp,
            disp_dense,
            nl,
            raman,
            gain,
            sheet,
            plugins,
            gain2,
            herm,
            tensor,
            dft,
            time,
            energy,
            dummy,
            device_name,
            ordinal,
            block: (bx, by),
            fused: fused_state,
            split_tab,
            cur_psi: 0,
            cplx,
            bsign,
            imag_sel: false,
            rev: None,
            reversing: false,
            inj_step: 0,
            dft_groups,
            inj_rounds: std::cell::RefCell::new(None),
        })
    }

    pub(crate) fn ensure_reverse(&mut self) -> Result<()> {
        if self.fused.is_some() {
            anyhow::bail!("time reversal runs the split kernels: build the backend with CudaBackend::new_with(plan, gpu, false)");
        }
        let p = &self.plan;
        if p.disp.is_some() || p.nonlin.is_some() || p.tensor.is_some() || p.sheet.is_some() || p.plugin.is_some() || p.gain2.is_some() || p.herm.is_some() {
            anyhow::bail!("time reversal supports non-dispersive, linear, diagonal media (and no sheets)");
        }
        if self.rev.is_some() {
            return Ok(());
        }
        self.rev = Some(Box::new(reverse_coefs(&self.plan, &self.stream)?));
        Ok(())
    }

    pub(crate) fn swap_reverse_e(&mut self) {
        self.reversing = !self.reversing;
        let r = self.rev.as_mut().unwrap();
        std::mem::swap(&mut self.cb, &mut r.cb);
        std::mem::swap(&mut self.ca, &mut r.ca);
    }

    pub(crate) fn swap_reverse_h(&mut self) {
        let r = self.rev.as_mut().unwrap();
        std::mem::swap(&mut self.ch, &mut r.ch);
        std::mem::swap(&mut self.hloss, &mut r.hloss);
        self.plan.ch_scalar = -self.plan.ch_scalar;
    }
}

/// Load every kernel of the step from the module.
fn load_kernels(module: &Arc<CudaModule>) -> Result<Kernels> {
    let f = |n: &str| module.load_function(n).with_context(|| format!("kernel {n}"));
    let k = Kernels {
        update_h: f("update_h")?,
        update_e: f("update_e")?,
        pml_e: f("pml_e")?,
        pml_h: f("pml_h")?,
        inject: f("inject")?,
        inject_complex: f("inject_complex")?,
        dft: f("dft_accumulate")?,
        gather: f("gather_box")?,
        scale_copy: f("scale_copy")?,
        disp_update: f("disp_update")?,
        disp_apply: f("disp_apply")?,
        hloss_scale: f("hloss_scale")?,
        disp_dense: f("disp_dense")?,
        nl_save: f("nl_save")?,
        nl_apply: f("nl_apply")?,
        raman_save: f("raman_save")?,
        raman_apply: f("raman_apply")?,
        gyro_apply: f("gyro_apply")?,
        gain_pre: f("gain_pre")?,
        gain_post: f("gain_post")?,
        tensor_dd: f("tensor_dd")?,
        tensor_apply: f("tensor_apply")?,
        column_energy: f("column_energy")?,
        step_fused: f("step_fused")?,
        step_fused_bloch: f("step_fused_bloch")?,
        e_plane: f("e_plane_fused")?,
        step2d: f("step2d")?,
        step2d_bloch: f("step2d_bloch")?,
        dft_boxes: f("dft_boxes")?,
        inject_lists: f("inject_lists")?,
    };
    Ok(k)
}

/// The CPML slabs: coefficient profiles and the four auxiliary fields of each.
fn pml_dev(s: &Arc<CudaStream>, plan: &Plan) -> Result<Vec<PmlDev>> {
    let mut pml = vec![];
    for p in &plan.pml {
        let c = p.cells();
        pml.push(PmlDev {
            axis: p.axis,
            start: p.start,
            size: p.size,
            e_b: dbuf(s, &p.e_b)?,
            e_a: dbuf(s, &p.e_a)?,
            e_k: dbuf(s, &p.e_kinv)?,
            h_b: dbuf(s, &p.h_b)?,
            h_a: dbuf(s, &p.h_a)?,
            h_k: dbuf(s, &p.h_kinv)?,
            psi: [zbuf(s, c)?, zbuf(s, c)?, zbuf(s, c)?, zbuf(s, c)?],
        });
    }
    Ok(pml)
}

impl InjDev {
    /// One list of source injections on the device.
    pub(crate) fn upload(s: &Arc<CudaStream>, j: &lightsim_core::plan::Injection) -> Result<InjDev> {
        Ok(InjDev {
            comp: j.comp,
            n: j.idx.len(),
            idx: dbuf(s, &j.idx)?,
            coef: dbuf(s, &j.coef)?,
            coef_im: match &j.coef_im {
                Some(c) => Some(dbuf(s, c)?),
                None => None,
            },
            group: dbuf(s, &j.group)?,
            amps: dbuf(s, &j.amps)?,
            ngroups: j.ngroups,
            lo: j.idx.iter().copied().min().unwrap_or(0),
            hi: j.idx.iter().copied().max().unwrap_or(0),
        })
    }
}

/// The lists of source injections on the device (empty lists dropped).
pub(crate) fn injections_dev(s: &Arc<CudaStream>, v: &[lightsim_core::plan::Injection]) -> Result<Vec<InjDev>> {
    v.iter().filter(|j| !j.idx.is_empty()).map(|j| InjDev::upload(s, j)).collect()
}

/// The amplitude tables (a_im, -a_re) of the complex sources of a list, which
/// drive the imaginary field set.
fn swapped_amps(s: &Arc<CudaStream>, v: &Vec<lightsim_core::plan::Injection>) -> Result<Vec<Option<DeviceBuffer<f32>>>> {
    v.iter()
        .filter(|j| !j.idx.is_empty())
        .map(|j| {
            if j.coef_im.is_none() {
                return Ok(None);
            }
            let mut t = j.amps.clone();
            for pr in t.chunks_mut(2) {
                let (ar, ai) = (pr[0], pr[1]);
                pr[0] = ai;
                pr[1] = -ar;
            }
            Ok(Some(dbuf(s, &t)?))
        })
        .collect()
}

/// The imaginary field set of a Bloch plan (fields, CPML and pole auxiliaries,
/// DFT accumulators, swapped source amplitudes).
fn imag_set(s: &Arc<CudaStream>, plan: &Plan) -> Result<Option<Box<CudaImag>>> {
    let swap_amps = |v: &Vec<lightsim_core::plan::Injection>| swapped_amps(s, v);
    let cplx = if plan.bloch {
        let len = plan.storage_len();
        let mut psi = vec![];
        for p in &plan.pml {
            let c = p.cells();
            psi.push([zbuf(s, c)?, zbuf(s, c)?, zbuf(s, c)?, zbuf(s, c)?]);
        }
        let mut disp = vec![];
        if let Some(d) = &plan.disp {
            for c in &d.comps {
                let n = c.idx.len();
                disp.push([zbuf(s, n * d.npole)?, zbuf(s, n * d.npole)?, zbuf(s, n)?]);
            }
        }
        let mut dftv = vec![];
        for d in &plan.dft {
            let nf = d.freqs.len();
            let mut bxs = vec![];
            for b in &d.boxes {
                let n = b.lcells() * nf;
                bxs.push((zbuf(s, n)?, zbuf(s, n)?));
            }
            dftv.push(bxs);
        }
        Some(Box::new(CudaImag {
            e: [zbuf(s, len)?, zbuf(s, len)?, zbuf(s, len)?],
            h: [zbuf(s, len)?, zbuf(s, len)?, zbuf(s, len)?],
            psi,
            disp,
            dft: dftv,
            amps_h: swap_amps(&plan.inj_h)?,
            amps_e: swap_amps(&plan.inj_e)?,
        }))
    } else {
        None
    };
    Ok(cplx)
}

/// The material table of the split E update in place of the per-cell
/// coefficient arrays (`LIGHTSIM_SPLIT_TAB=0` keeps the arrays).
fn split_table(s: &Arc<CudaStream>, plan: &Plan, fused: bool) -> Result<Option<(DeviceBuffer<u8>, DeviceBuffer<f32>, u32)>> {
    let split_tab = if !fused && std::env::var("LIGHTSIM_SPLIT_TAB").as_deref() != Ok("0") {
        let (mat, tab) = fused::material_table(plan);
        let n = tab.len() / 6;
        if n <= 65536 {
            let (bytes, mode) = fused::pack_index(mat, n);
            Some((dbuf(s, &bytes)?, dbuf(s, &tab)?, mode))
        } else {
            None
        }
    } else {
        None
    };
    Ok(split_tab)
}

/// The dense layout of dispersive media in a fused run (`LIGHTSIM_DISP_DENSE`:
/// unset = automatic, at least 25 % of the cells dispersive; "1" = always; "0"
/// = never).
fn disp_dense_dev(s: &Arc<CudaStream>, plan: &Plan, fused_state: &Option<fused::FusedState>) -> Result<Option<DispDenseDev>> {
    let dense_env = std::env::var("LIGHTSIM_DISP_DENSE").ok();
    let disp_dense = match (fused_state, &plan.disp) {
        // (the 2-D step takes the dispersive cells from the lists)
        (Some(fs), Some(d))
            if fs.twod.is_none()
                && dense_env.as_deref() != Some("0")
                && (dense_env.as_deref() == Some("1")
                    || d.comps.iter().map(|c| c.idx.len()).sum::<usize>() * 4 >= 3 * plan.local_cells())
                && d.coefs.len() / (3 * d.npole.max(1)) < 255 =>
        {
            let len = plan.storage_len();
            let nmat = d.coefs.len() / (3 * d.npole.max(1));
            let mut comps = vec![];
            for c in &d.comps {
                // (mixed voxels of smoothed dispersive dielectrics have a scale of
                // their own: they stay in the sparse lists, see `disp_mixed` below)
                let is_mixed = |t: usize| c.mixed.binary_search(&(t as u32)).is_ok();
                let mut scale = vec![0f32; nmat.max(1)];
                for (t, &m) in c.matid.iter().enumerate() {
                    if !is_mixed(t) {
                        scale[m as usize] = c.scale[t];
                    }
                }
                // every interior cell in the same dispersive material
                let uniform = c.mixed.is_empty()
                    && c.idx.len() == plan.local_cells()
                    && !c.matid.is_empty()
                    && c.matid.iter().all(|&m| m == c.matid[0])
                    && std::env::var("LIGHTSIM_DISP_UNIFORM").as_deref() != Ok("0");
                let (dm, dmu) = if uniform {
                    (vec![0u8; 4], c.matid[0] + 1)
                } else {
                    let mut dm = vec![0u8; len];
                    for (t, &si) in c.idx.iter().enumerate() {
                        if !is_mixed(t) {
                            dm[si as usize] = c.matid[t] as u8 + 1;
                        }
                    }
                    (dm, 0)
                };
                comps.push((c.comp.axis(), dbuf(s, &dm)?, zbuf(s, d.npole * len)?, zbuf(s, d.npole * len)?, dbuf(s, &scale)?, dmu));
            }
            Some(DispDenseDev { npole: d.npole, coefs: dbuf(s, &d.coefs)?, comps })
        }
        _ => None,
    };
    Ok(disp_dense)
}

/// The sparse lists of dispersive cells: all of them, or with the dense layout
/// only the mixed voxels of smoothed dispersive dielectrics (each cell is in
/// one of the two; both run in the fused step).
fn disp_dev(s: &Arc<CudaStream>, plan: &Plan, dense: bool) -> Result<Option<DispDev>> {
    let mixed_only = |c: &lightsim_core::plan::DispComponent| -> lightsim_core::plan::DispComponent {
        let pick = |v: &[u64]| c.mixed.iter().map(|&t| v[t as usize]).collect::<Vec<_>>();
        lightsim_core::plan::DispComponent {
            comp: c.comp,
            idx: pick(&c.idx),
            matid: c.mixed.iter().map(|&t| c.matid[t as usize]).collect(),
            scale: c.mixed.iter().map(|&t| c.scale[t as usize]).collect(),
            mixed: (0..c.mixed.len() as u32).collect(),
        }
    };
    let sparse_comps: Option<Vec<lightsim_core::plan::DispComponent>> = match (dense, &plan.disp) {
        (false, Some(d)) => Some(d.comps.clone()),
        (true, Some(d)) if d.comps.iter().any(|c| !c.mixed.is_empty()) => Some(d.comps.iter().map(mixed_only).filter(|c| !c.idx.is_empty()).collect()),
        _ => None,
    };
    let disp = match (&plan.disp, sparse_comps) {
        (Some(d), Some(sc)) => {
            let mut comps = vec![];
            for c in &sc {
                let n = c.idx.len();
                comps.push(DispCompDev {
                    comp: c.comp,
                    n,
                    idx: dbuf(s, &c.idx)?,
                    matid: dbuf(s, &c.matid)?,
                    scale: dbuf(s, &c.scale)?,
                    p: zbuf(s, n * d.npole)?,
                    pprev: zbuf(s, n * d.npole)?,
                    dp: zbuf(s, n)?,
                });
            }
            Some(DispDev { npole: d.npole, coefs: dbuf(s, &d.coefs)?, comps })
        }
        _ => None,
    };
    Ok(disp)
}

/// The chi2 / chi3 cell lists.
fn nonlinear_dev(s: &Arc<CudaStream>, plan: &Plan) -> Result<Vec<NlCompDev>> {
    let mut nl = vec![];
    if let Some(np) = &plan.nonlin {
        for c in &np.comps {
            let n = c.idx.len();
            nl.push(NlCompDev { comp: c.comp, n, idx: dbuf(s, &c.idx)?, c2: dbuf(s, &c.c2)?, c3: dbuf(s, &c.c3)?, prev: zbuf(s, n)? });
        }
    }
    Ok(nl)
}

/// The chi3 Raman lists and coefficient table.
fn raman_dev(s: &Arc<CudaStream>, plan: &Plan) -> Result<Option<(DeviceBuffer<f32>, Vec<RamanCompDev>)>> {
    let raman = match plan.nonlin.as_ref().and_then(|n| n.raman.as_ref()) {
        Some(r) => {
            let mut comps = vec![];
            for c in &r.comps {
                let n = c.idx.len();
                comps.push(RamanCompDev {
                    comp: c.comp,
                    n,
                    idx: dbuf(s, &c.idx)?,
                    matid: dbuf(s, &c.matid)?,
                    c2: dbuf(s, &c.c2)?,
                    c3: dbuf(s, &c.c3)?,
                    cr: dbuf(s, &c.cr)?,
                    prev: zbuf(s, n)?,
                    q: zbuf(s, n)?,
                    qp: zbuf(s, n)?,
                });
            }
            Some((dbuf(s, &r.coefs)?, comps))
        }
        None => None,
    };
    Ok(raman)
}

/// The four-level gain lists, their populations and the medium table.
fn gain_dev(s: &Arc<CudaStream>, plan: &Plan) -> Result<Option<(DeviceBuffer<f64>, Vec<GainCompDev>)>> {
    let gain = match plan.nonlin.as_ref().and_then(|n| n.gain.as_ref()) {
        Some(g) => {
            let mut comps = vec![];
            for c in &g.comps {
                let n = c.idx.len();
                let mut pop = vec![0f64; 4 * n];
                for (t, &r) in c.row.iter().enumerate() {
                    for l in 0..4 {
                        pop[l * n + t] = g.init[4 * r as usize + l];
                    }
                }
                comps.push(GainCompDev {
                    comp: c.comp,
                    n,
                    idx: dbuf(s, &c.idx)?,
                    row: dbuf(s, &c.row)?,
                    scale: dbuf(s, &c.scale)?,
                    vol: c.vol.clone(),
                    prev: zbuf(s, n)?,
                    p: zbuf(s, n)?,
                    pp: zbuf(s, n)?,
                    dp: zbuf(s, n)?,
                    pop: dbuf(s, &pop)?,
                });
            }
            Some((dbuf(s, &g.tab)?, comps))
        }
        None => None,
    };
    Ok(gain)
}

/// The off-diagonal, oriented and gyrotropic lists.
fn tensor_dev(s: &Arc<CudaStream>, plan: &Plan) -> Result<Option<TensorDev>> {
    let tensor = match &plan.tensor {
        Some(t) => {
            let n = (t.p1 - t.p0) * plan.plane();
            let mut comps = vec![];
            let mut ops = vec![];
            for c in &t.comps {
                comps.push((c.comp, c.idx.len(), dbuf(s, &c.idx)?, dbuf(s, &c.c1)?, dbuf(s, &c.c2)?, dbuf(s, &c.m1)?, dbuf(s, &c.m2)?));
                let ns = c.idx.len() * t.op_np;
                ops.push((dbuf(s, &c.opm)?, dbuf(s, &c.oscale)?, zbuf(s, ns)?, zbuf(s, ns)?));
            }
            let optab = dbuf(s, &t.op_tab)?;
            let mut gyro = vec![];
            for g in &t.gyro {
                let ns = g.idx.len() * 3 * t.gyro_np;
                gyro.push((g.comp.axis(), g.idx.len(), dbuf(s, &g.idx)?, dbuf(s, &g.row)?, dbuf(s, &g.scale)?, zbuf(s, ns)?, zbuf(s, ns)?));
            }
            // (a plan without gyrotropic media gets a one-float dummy table)
            let gyro_tab = if t.gyro_tab.is_empty() { zbuf(s, 1)? } else { dbuf(s, &t.gyro_tab)? };
            Some(TensorDev { prev: [zbuf(s, n)?, zbuf(s, n)?, zbuf(s, n)?], en: [zbuf(s, n)?, zbuf(s, n)?, zbuf(s, n)?], comps, ops, optab, gyro, gyro_tab })
        }
        None => None,
    };
    Ok(tensor)
}

/// The DFT accumulators and phasor tables of every frequency-domain monitor.
fn dft_dev(s: &Arc<CudaStream>, plan: &Plan) -> Result<Vec<DftDev>> {
    let mut dft = vec![];
    for d in &plan.dft {
        let nf = d.freqs.len();
        let mut boxes = vec![];
        for b in &d.boxes {
            let n = b.lcells() * nf;
            boxes.push(DftBoxDev {
                local: b.is_local(),
                comp: b.comp,
                lstart: b.lstart,
                lsize: b.lsize,
                re: zbuf(s, n)?,
                im: zbuf(s, n)?,
            });
        }
        dft.push(DftDev { nf, stride: d.stride, phasor_e: dbuf(s, &d.phasor_e)?, phasor_h: dbuf(s, &d.phasor_h)?, boxes });
    }
    Ok(dft)
}

/// The time monitors; their recordings are allocated as the run reaches them.
fn time_dev(plan: &Plan) -> Vec<TimeDev> {
    let mut time = vec![];
    for t in &plan.time {
        let count = if plan.nsteps > t.start_step { (plan.nsteps - t.start_step).div_ceil(t.interval) } else { 0 };
        let mut boxes = vec![];
        for b in &t.boxes {
            boxes.push(TimeBoxDev {
                local: b.is_local(),
                comp: b.comp,
                lstart: b.lstart,
                lsize: b.lsize,
                chunk: TimeBoxDev::chunk_for(b.lcells(), count),
                bufs: vec![],
            });
        }
        time.push(TimeDev { interval: t.interval, start: t.start_step, count, steps: vec![], boxes });
    }
    time
}
