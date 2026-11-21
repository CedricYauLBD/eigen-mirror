## Eigen CI infrastructure

Eigen's CI infrastructure uses three stages:
  1. A `checkformat` stage to verify MRs satisfy proper formatting style, as
     defined by `clang-format`.
  2. A `build` stage to build the unit-tests.
  3. A `test` stage to run the unit-tests.

For merge requests, only a small subset of tests are built/run, and only on a
small subset of platforms.  This is to reduce our overall testing infrastructure
resource usage.  In addition, a weekly scheduled pipeline builds and runs the
full suite of tests on most officially supported platforms.

## Persistent compiler cache

Self-hosted runners can configure a persistent host directory for ccache to
avoid the 5 GB GitLab archive limit and eliminate compression overhead.
Set the standard `CCACHE_*` environment variables in the runner's `config.toml`:

```toml
[[runners]]
  environment = [
    "CCACHE_DIR=/ccache",
    "CCACHE_MAXSIZE=50G",
  ]
  [runners.docker]
    volumes = ["/var/cache/eigen-ccache:/ccache:rw", "/cache"]
```

The YAML templates prefix their defaults with `EIGEN_CI_CCACHE_*`
(`EIGEN_CI_CCACHE_DIR`, `EIGEN_CI_CCACHE_MAXSIZE`, `EIGEN_CI_CCACHE_BASEDIR`,
`EIGEN_CI_CCACHE_COMPRESSLEVEL`) so that runner-level `environment = [...]`
settings in `config.toml` are not shadowed by GitLab CI.  When a runner sets
standard `CCACHE_DIR`, the build scripts (`build.linux.script.sh` and
`build.windows.script.ps1`) preserve the runner's value, leaving
`${CI_PROJECT_DIR}/.ccache` absent so GitLab's `restore_cache` and
`archive_cache` steps are no-ops.

If the runner has already cached these jobs locally, `restore_cache` still
extracts the stale pool into `${CI_PROJECT_DIR}/.ccache` on every job (and
`archive_cache` re-archives it whenever a job's primary key has no archive yet,
such as the first run under a new `-mr<iid>` key).  When switching an existing
runner to a host `CCACHE_DIR`, clear the runner's local cache storage once (the
`/cache` volumes, or `clear-docker-cache` for the Docker executor); once
`.ccache/` is absent, `cache-archiver` reports `No files to cache` and nothing
recreates the archive.  A runner with a distributed `[runners.cache]` backend
has no equivalent one-time clear—any other runner writing the same bucket
recreates the archive—so a host directory only fits cleanly on runners with a
local-only cache, where it also costs the shared pool nothing.

Concurrent jobs may share one local directory on a POSIX or NTFS filesystem,
but not over network shares.  The build scripts record per-job cache hits and
misses via `CCACHE_STATSLOG` and `ccache --show-log-stats` so that concurrent
jobs sharing a cache directory do not zero or mix each other's counters.

## MSVC coverage for merge requests

The native Windows jobs need the dedicated Windows runners and run only under a
label, so merge requests get their default MSVC coverage from
`build:windows:cross:...`, which cross-compiles the smoke tests on a hosted
Linux runner by running the real MSVC toolchain under Wine, and
`test:windows:cross:...`, which runs the resulting binaries on a hosted Windows
runner.  Both use hosted runners and pre-baked images, so they also run in
forks.

The toolchain is not part of the CI image (see
`ci/docker/ubuntu-24.04-amd64-msvc-wine-build/`).  The [Visual Studio Community
2022 license](https://visualstudio.microsoft.com/license-terms/vs2022-ga-community/)
permits using MSVC to develop and test software under an OSI-approved license,
Eigen's MPL-2.0 among them, and permits installing Build Tools into a container
"dedicated solely to your use"; it also says you may not "share, publish, rent
or lease the software", which a registry image would be.  So it is
downloaded from Microsoft's installer manifests by
`ci/scripts/setup.msvc-wine.sh` on each run and never lands in an image layer or
a job artifact.  Nothing is licensed or activated at run time: the compiler
builds with no network access once installed.
