//! The kernels' view of the boundaries: codes of the y and z sides, Bloch
//! phases and how the x ghost planes of this slab are filled.

use super::*;

pub(super) struct BoundaryCodes {
    pub bc_y: [u32; 2],
    pub bc_z: [u32; 2],
    /// (cos, sin) of k L per axis; (1, 0) without a Bloch boundary
    pub bloch_phase: [(f64, f64); 3],
    /// a Bloch boundary with k != 0 along y or z
    pub bloch_yz: bool,
    /// a Bloch boundary with k != 0 along x
    pub bloch_x: bool,
    pub ghost_e_hi: GhostRule,
    pub ghost_h_lo: GhostRule,
}

impl BoundaryCodes {
    /// Complex fields on every slab: some axis is Bloch with k != 0.
    pub fn complex_fields(&self) -> bool {
        self.bloch_x || self.bloch_yz
    }
}

pub(super) fn compile(s: &Setup) -> Result<BoundaryCodes> {
    let (b, grid, opts) = (&s.sim.boundaries, s.grid, s.opts);
    let nx = s.n[0];
    let code_lo = |bd: &Boundary| -> Result<u32> {
        Ok(match bd {
            Boundary::Periodic => BC_PERIODIC,
            Boundary::Bloch { k } => {
                if *k != 0.0 { BC_BLOCH } else { BC_PERIODIC }
            }
            Boundary::Pmc => BC_MIRROR,
            _ => BC_ZERO,
        })
    };
    let bc_y = [code_lo(&b.y[0])?, code_lo(&b.y[1])?];
    let bc_z = [code_lo(&b.z[0])?, code_lo(&b.z[1])?];
    let mut bloch_phase = [(1.0f64, 0.0f64); 3];
    for a in 0..3 {
        if let Boundary::Bloch { k } = &b.axis(a)[0] {
            let e = grid.edges(a);
            let ph = k * (e[e.len() - 1] - e[0]);
            bloch_phase[a] = (ph.cos(), ph.sin());
        }
    }
    let bloch_yz = bc_y[0] == BC_BLOCH || bc_z[0] == BC_BLOCH;
    // complex fields on every slab when any axis is Bloch with k != 0
    let bloch_x = matches!(b.x[0], Boundary::Bloch { k } if k != 0.0);

    // ---- x ghost rules ----
    let first = opts.slab_index == 0;
    let last = opts.slab_index + 1 == opts.nslabs.max(1);
    let lx = grid.x[nx] - grid.x[0];
    let ghost_e_hi = if !last {
        GhostRule::Neighbor
    } else {
        match &b.x[1] {
            Boundary::Periodic => {
                if opts.nslabs > 1 { GhostRule::Neighbor } else { GhostRule::Periodic { cos: 1.0, sin: 0.0 } }
            }
            Boundary::Bloch { k } => {
                // several slabs: the chain's wrap exchange applies the phase
                let ph = k * lx;
                if opts.nslabs > 1 { GhostRule::Neighbor } else { GhostRule::Periodic { cos: ph.cos(), sin: ph.sin() } }
            }
            Boundary::Pmc => GhostRule::Mirror { factor: 1.0 },
            _ => GhostRule::Zero,
        }
    };
    let ghost_h_lo = if !first {
        GhostRule::Neighbor
    } else {
        match &b.x[0] {
            Boundary::Periodic => {
                if opts.nslabs > 1 { GhostRule::Neighbor } else { GhostRule::Periodic { cos: 1.0, sin: 0.0 } }
            }
            Boundary::Bloch { k } => {
                let ph = -k * lx;
                if opts.nslabs > 1 { GhostRule::Neighbor } else { GhostRule::Periodic { cos: ph.cos(), sin: ph.sin() } }
            }
            Boundary::Pmc => GhostRule::Mirror { factor: -1.0 },
            _ => GhostRule::Zero,
        }
    };
    Ok(BoundaryCodes { bc_y, bc_z, bloch_phase, bloch_yz, bloch_x, ghost_e_hi, ghost_h_lo })
}
