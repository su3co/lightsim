//! lightsim's own CPU thread pool.
//!
//! The parallel work of the engines, the set-up and the solvers runs in a
//! rayon pool that belongs to lightsim rather than in rayon's global pool,
//! and that pool is keyed by the process id. A process forked from one that
//! had already used the pool has none of its worker threads (fork copies only
//! the calling thread), so jobs sent to the inherited pool would never run and
//! the child would hang; here the child sees a new process id and builds a
//! fresh pool. The parent's pool is left alone in the child (never dropped:
//! its threads do not exist there).
//!
//! The pool has as many threads as rayon's global pool would have:
//! `RAYON_NUM_THREADS` when set, else one per logical CPU; each has an 8 MiB
//! stack, since the sequential part of every call runs on one of them too.
//!
//! Entry points (the Python bindings, the CLI) run their work inside
//! [`install`], and every thread lightsim starts itself (one per device in a
//! multi-device run) does the same, so that no work reaches rayon's global pool.

use rayon::{ThreadPool, ThreadPoolBuilder};
use std::sync::atomic::{AtomicPtr, Ordering};

struct Keyed {
    pid: u32,
    pool: ThreadPool,
}

static POOL: AtomicPtr<Keyed> = AtomicPtr::new(std::ptr::null_mut());

/// The pool of this process (built on first use, and again after a fork).
pub fn pool() -> &'static ThreadPool {
    let pid = std::process::id();
    let cur = POOL.load(Ordering::Acquire);
    // SAFETY: published pools are leaked, never freed
    if let Some(k) = unsafe { cur.as_ref() }
        && k.pid == pid
    {
        return &k.pool;
    }
    let pool = ThreadPoolBuilder::new()
        .thread_name(|i| format!("lightsim-{i}"))
        // (entry points now run on a worker: give it a main thread's stack)
        .stack_size(8 << 20)
        .build()
        .expect("lightsim: cannot start the CPU thread pool");
    let new = Box::into_raw(Box::new(Keyed { pid, pool }));
    match POOL.compare_exchange(cur, new, Ordering::AcqRel, Ordering::Acquire) {
        // SAFETY: just published, never freed
        Ok(_) => unsafe { &(*new).pool },
        Err(won) => {
            // another thread of this process published one first: use it
            // SAFETY: `new` was never shared; `won` is leaked like every published pool
            drop(unsafe { Box::from_raw(new) });
            let k = unsafe { &*won };
            if k.pid == pid { &k.pool } else { self::pool() }
        }
    }
}

/// Run `f` on lightsim's pool. A caller that is already a worker of a rayon
/// pool (lightsim's, or one of its own) runs `f` in place.
pub fn install<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    if rayon::current_thread_index().is_some() {
        return f();
    }
    pool().install(f)
}

#[cfg(test)]
mod tests {
    #[test]
    fn same_pool_and_inline_when_nested() {
        let a = super::pool() as *const _;
        assert_eq!(a, super::pool() as *const _);
        let n = super::install(|| (rayon::current_num_threads(), super::install(rayon::current_thread_index)));
        assert_eq!(n.0, super::pool().current_num_threads());
        assert!(n.1.is_some());
    }
}
