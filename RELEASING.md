# Releasing lightsim

A release is a tag `v<version>` pushed to GitHub. The workflow
`.github/workflows/workflow.yml` then builds every wheel and the sdist, tests
them, checks that the tag matches the package version and publishes to PyPI
through trusted publishing (OpenID Connect from the GitHub environment `pypi`;
there is no token or secret). Nothing else publishes.

## The version, in one place

The version is `version` under `[workspace.package]` in `Cargo.toml`. The
crates inherit it, maturin copies it into the Python package metadata
(`pyproject.toml` declares it `dynamic`), and `lightsim.__version__` and
`lightsim info` report it. Change it there only, then refresh the lock file:

```bash
cargo update --workspace
```

## Before tagging

1. `main` holds everything for the release, and the `wheels` and `rust`
   workflows are green on its last commit (every wheel built, installed in a
   clean environment and tested on its platform; the sdist built from source).
2. The GPU path, which CI cannot run: on a machine with an NVIDIA GPU, install
   the Linux wheel from that run's artifacts in a fresh virtual environment
   and run `python .github/scripts/smoke_test.py --gpu`, and run
   `lightsim validate --device cuda:0` from a source build of the same commit.
3. `docs/guide/install.md` describes what is being released (platforms, GPU
   requirements); `python docs/check_samples.py` passes.
4. The version in `Cargo.toml` is the one to release and has never been
   uploaded to PyPI before.

## Tag and publish

```bash
git checkout main && git pull
git tag -a v0.1.0 -m "lightsim 0.1.0"
git push origin v0.1.0
```

Follow the run under Actions. The job "publish to PyPI" waits for every build
and test job and for the tag check; if the `pypi` environment has required
reviewers, approve the deployment there. A tag that does not match the version
stops the run before anything is uploaded: delete the tag
(`git push origin :refs/tags/v0.1.0`), fix, and tag again.

## After publishing

- https://pypi.org/project/lightsim/ shows the new version with five wheels
  (`manylinux_2_28_x86_64`, `manylinux_2_28_aarch64`, `macosx_11_0_arm64`,
  `macosx_10_13_x86_64`, all `cp310-abi3`) and the sdist, the README and the
  project links, and attestations for each file.
- In a fresh virtual environment: `pip install lightsim==<version>`, then
  `python -c "import lightsim; print(lightsim.__version__, lightsim.devices())"`;
  on a GPU machine the GPUs are listed.
- Create the GitHub release for the tag, if wanted, and deploy the
  documentation site so the install page goes live with the package.

## A release cannot be replaced

PyPI never accepts a file name twice: a version once uploaded cannot be
re-uploaded or overwritten, even after it is deleted. A broken release is
**yanked** on PyPI (Manage project, Releases, Options, Yank): pip then skips
it unless it is pinned exactly. Fix the problem and release the next version
(`0.1.1`).
