//! TCP point-to-point links for multi-node runs (direct point-to-point links
//! between neighbouring machines or a shared Ethernet network).
//!
//! Ranks are arranged in a chain along x (optionally closed into a ring for
//! periodic x boundaries). Rank `r` listens on `base_port + r` and connects to
//! rank `r + 1`'s listener, so every link is set up without races.

use crate::Real;
use crate::run::Link;
use anyhow::{Context, Result, bail};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

pub struct TcpLink {
    stream: TcpStream,
    bytes: Vec<u8>,
}

impl TcpLink {
    pub fn new(stream: TcpStream) -> Result<TcpLink> {
        stream.set_nodelay(true)?;
        Ok(TcpLink { stream, bytes: Vec::new() })
    }
}

impl Link for TcpLink {
    fn send(&mut self, data: &[Real]) -> Result<()> {
        // float slices are plain little-endian on all supported targets
        let b = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
        self.stream.write_all(b)?;
        Ok(())
    }
    fn recv(&mut self, out: &mut [Real]) -> Result<()> {
        let b = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, std::mem::size_of_val(out)) };
        self.stream.read_exact(b)?;
        let _ = &self.bytes;
        Ok(())
    }
}

/// Send/receive an arbitrary byte message with a u64 length prefix.
pub fn send_bytes(s: &mut TcpStream, data: &[u8]) -> Result<()> {
    s.write_all(&(data.len() as u64).to_le_bytes())?;
    s.write_all(data)?;
    Ok(())
}

pub fn recv_bytes(s: &mut TcpStream) -> Result<Vec<u8>> {
    let mut l = [0u8; 8];
    s.read_exact(&mut l)?;
    let n = u64::from_le_bytes(l) as usize;
    let mut v = vec![0u8; n];
    s.read_exact(&mut v)?;
    Ok(v)
}

fn connect_retry(addr: &str, timeout: Duration) -> Result<TcpStream> {
    let t0 = Instant::now();
    loop {
        match TcpStream::connect(addr) {
            Ok(s) => return Ok(s),
            Err(e) => {
                if t0.elapsed() > timeout {
                    return Err(e).with_context(|| format!("connecting to {addr}"));
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

/// Links of one rank in a chain/ring of `hosts.len()` ranks.
pub struct RankLinks {
    pub lo: Option<TcpLink>,
    pub hi: Option<TcpLink>,
    pub lo_is_wrap: bool,
    pub hi_is_wrap: bool,
    /// Control connection to rank 0 (ranks > 0) or to every other rank (rank 0).
    pub control: Vec<TcpStream>,
}

/// Establish the chain (and, if `ring`, the wrap-around link) plus a control
/// star centred on rank 0. `hosts[r]` is the address other ranks use to reach
/// rank `r`. `chain_addrs[r]` (optional, length nranks-1) overrides the
/// address rank `r` uses to reach rank `r+1` for the halo link, e.g. a direct
/// point-to-point link between two machines (e.g. a direct 200 GbE cable).
pub fn connect(rank: usize, hosts: &[String], base_port: u16, ring: bool) -> Result<RankLinks> {
    connect_with(rank, hosts, None, base_port, ring)
}

pub fn connect_with(rank: usize, hosts: &[String], chain_addrs: Option<&[String]>, base_port: u16, ring: bool) -> Result<RankLinks> {
    let nr = hosts.len();
    if rank >= nr {
        bail!("rank {rank} out of range for {nr} hosts");
    }
    let timeout = Duration::from_secs(120);
    let listener = TcpListener::bind(("0.0.0.0", base_port + rank as u16))
        .with_context(|| format!("binding port {}", base_port + rank as u16))?;
    let mut lo = None;
    let mut hi = None;
    let mut lo_is_wrap = false;
    let mut hi_is_wrap = false;
    let mut control = Vec::new();
    // Each rank first sends a 2-byte hello (kind, from) on every connection it opens.
    // kind 0 = chain (from r-1 to r), 1 = wrap (from last to 0), 2 = control (to rank 0).
    let mut expected = 0usize;
    if rank > 0 {
        expected += 1; // chain from rank-1
    }
    if ring && nr > 1 && rank == 0 {
        expected += 1; // wrap from last
    }
    if rank == 0 {
        expected += nr - 1; // control
    }
    // outgoing
    if rank + 1 < nr {
        let addr = chain_addrs.and_then(|c| c.get(rank)).unwrap_or(&hosts[rank + 1]);
        let mut s = connect_retry(&format!("{}:{}", addr, base_port + rank as u16 + 1), timeout)?;
        s.write_all(&[0u8, rank as u8])?;
        hi = Some(TcpLink::new(s)?);
    }
    if ring && nr > 1 && rank == nr - 1 {
        let mut s = connect_retry(&format!("{}:{}", hosts[0], base_port), timeout)?;
        s.write_all(&[1u8, rank as u8])?;
        hi = Some(TcpLink::new(s)?);
        hi_is_wrap = true;
    }
    if rank > 0 {
        let mut s = connect_retry(&format!("{}:{}", hosts[0], base_port), timeout)?;
        s.write_all(&[2u8, rank as u8])?;
        s.set_nodelay(true)?;
        control.push(s);
    }
    // incoming
    let mut ctrl_in: Vec<(u8, TcpStream)> = Vec::new();
    for _ in 0..expected {
        let (mut s, _) = listener.accept()?;
        let mut hello = [0u8; 2];
        s.read_exact(&mut hello)?;
        match hello[0] {
            0 => lo = Some(TcpLink::new(s)?),
            1 => {
                lo = Some(TcpLink::new(s)?);
                lo_is_wrap = true;
            }
            _ => {
                s.set_nodelay(true)?;
                ctrl_in.push((hello[1], s));
            }
        }
    }
    ctrl_in.sort_by_key(|(r, _)| *r);
    control.extend(ctrl_in.into_iter().map(|(_, s)| s));
    Ok(RankLinks { lo, hi, lo_is_wrap, hi_is_wrap, control })
}
