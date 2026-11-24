//! Monitors on the device: DFT accumulation, the field energy and the
//! download of the recorded results.

use super::*;

impl CudaBackend {
    pub(crate) fn record_dft(&self, step: usize) -> Result<()> {
        if !self.dft_groups.is_empty() {
            return self.record_dft_groups(step);
        }
        let (ny, nz) = (self.plan.ny() as u32, self.plan.nz() as u32);
        for d in &self.dft {
            if step % d.stride != 0 {
                continue;
            }
            let row_off = (step * d.nf * 2 * 4) as u64;
            for b in &d.boxes {
                if !b.local || self.idle_2d(b.comp) {
                    continue;
                }
                let n = b.lsize[0] * b.lsize[1] * b.lsize[2];
                let ph = if b.comp.is_e() { d.phasor_e.cu_deviceptr() } else { d.phasor_h.cu_deviceptr() };
                let args = Args::new()
                    .p(self.new_ptr(b.comp))
                    .p(b.re.cu_deviceptr())
                    .p(b.im.cu_deviceptr())
                    .p(ph + row_off)
                    .u(d.nf as u32)
                    .u(b.lstart[0] as u32)
                    .u(b.lstart[1] as u32)
                    .u(b.lstart[2] as u32)
                    .u(b.lsize[0] as u32)
                    .u(b.lsize[1] as u32)
                    .u(b.lsize[2] as u32)
                    .u(ny)
                    .u(nz);
                self.launch_1d(&self.k.dft, n, args)?;
            }
        }
        Ok(())
    }

    pub(crate) fn energy_set(&self) -> Result<f64> {
        let p = &self.plan;
        let args = Args::new()
            .p(self.e[0].cu_deviceptr())
            .p(self.e[1].cu_deviceptr())
            .p(self.e[2].cu_deviceptr())
            .p(self.h[0].cu_deviceptr())
            .p(self.h[1].cu_deviceptr())
            .p(self.h[2].cu_deviceptr())
            .p(self.energy.cu_deviceptr())
            .u(p.nxl as u32)
            .u(p.ny() as u32)
            .u(p.nz() as u32);
        self.launch_1d(&self.k.column_energy, p.plane(), args)?;
        let v = self.energy.to_host_vec(&self.stream)?;
        let tot: f64 = v.iter().map(|x| *x as f64).sum();
        if tot.is_finite() {
            return Ok(tot);
        }
        // The kernel squares and sums in single precision, which overflows
        // for fields above about 1e19 (a dipole of unit moment in nanometre
        // cells) although the fields themselves are finite. Decide on the
        // host, in double precision: the result is then finite for finite
        // fields, and infinite or NaN only if the fields are.
        self.stream.synchronize()?;
        let plane = p.plane();
        let mut sum = 0.0f64;
        for buf in self.e.iter().chain(self.h.iter()) {
            let host = buf.to_host_vec(&self.stream)?;
            sum += host[plane..(p.nxl + 1) * plane].iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
        }
        Ok(sum)
    }

    pub(crate) fn download_set(&self) -> Result<RawResults> {
        self.stream.synchronize()?;
        let mut raw = RawResults::default();
        for d in &self.dft {
            let mut boxes = vec![];
            for b in &d.boxes {
                if b.local {
                    let n = b.lsize[0] * b.lsize[1] * b.lsize[2] * d.nf;
                    let mut re = b.re.to_host_vec(&self.stream)?;
                    let mut im = b.im.to_host_vec(&self.stream)?;
                    re.truncate(n);
                    im.truncate(n);
                    boxes.push((re, im));
                } else {
                    boxes.push((vec![], vec![]));
                }
            }
            raw.dft.push(RawDft { boxes });
        }
        for t in &self.time {
            let mut boxes = vec![];
            for b in &t.boxes {
                if b.local {
                    let cells = b.lsize[0] * b.lsize[1] * b.lsize[2];
                    let mut v = vec![0f32; cells * t.steps.len()];
                    for (c, part) in v.chunks_mut((b.chunk * cells).max(1)).enumerate() {
                        unsafe { memcpy_dtoh_async(part.as_mut_ptr(), b.bufs[c].cu_deviceptr(), part.len() * 4, self.stream.cu_stream()) }.map_err(|e| anyhow!("{e:?}"))?;
                    }
                    self.stream.synchronize()?;
                    boxes.push(v);
                } else {
                    boxes.push(vec![]);
                }
            }
            raw.time.push(RawTime { boxes, steps: t.steps.clone() });
        }
        Ok(raw)
    }
}
