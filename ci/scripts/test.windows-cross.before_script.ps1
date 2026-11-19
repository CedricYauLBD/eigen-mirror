# Reports the MSVC toolset the cross-compiled tests were built with.
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0
#
# test.windows.script.ps1 relocates the Linux paths CMake baked into the build
# tree, as it does for a tree built in another runner's checkout.

# Write-Information is preferred over Write-Host, but the information stream is
# discarded unless this is set: $InformationPreference defaults to
# SilentlyContinue, so the messages below would never reach the job log.
$InformationPreference = 'Continue'

# The binaries link the MSVC runtime dynamically, and that runtime has to be at
# least as new as the toolset they were built with.  Report both, so a run that
# fails because the two drifted apart says so instead of looking like 166 broken
# tests.  Nothing is enforced here: the runners are the moving part, and this
# job is allowed to fail.
$builtWith = 'unknown'
$toolsetStamp = Join-Path ${EIGEN_CI_BUILDDIR} '.eigen_ci_msvc_toolset'
if (Test-Path $toolsetStamp) {
  $builtWith = (Get-Content $toolsetStamp -First 1).Trim()
}

$runnerHas = 'unknown'
$runnerVs = $null
$vswhere = "${Env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vswhere) {
  # -products * is required: vswhere defaults to Community/Professional/
  # Enterprise, and the hosted runners install Build Tools, which it would
  # otherwise skip.
  $vsPath = & $vswhere -latest -products * -property installationPath
  if ($vsPath) {
    $versionFile = Join-Path $vsPath 'VC\Auxiliary\Build\Microsoft.VCToolsVersion.default.txt'
    if (Test-Path $versionFile) {
      $runnerHas = (Get-Content $versionFile -First 1).Trim()
    }
  }
  # EIGEN_CI_MSVC_VS_VERSION takes the Visual Studio product version, which is
  # what vswhere reports, so the fix can be quoted verbatim below.
  $installed = & $vswhere -latest -products * -property installationVersion
  if ($installed -match '^(\d+\.\d+)') { $runnerVs = $Matches[1] }
}

$verdict = if ($builtWith -eq 'unknown' -or $runnerHas -eq 'unknown') {
  ' (cannot compare)'
} elseif ($builtWith -eq $runnerHas) {
  ' (match)'
} else {
  ' (differ)'
}
Write-Information "MSVC toolset: built with ${builtWith}, this runner has ${runnerHas}${verdict}"

if ($builtWith -ne 'unknown' -and $runnerHas -ne 'unknown' -and $builtWith -ne $runnerHas) {
  Write-Warning "Build and runner toolset versions have drifted apart."
  if ([version]$builtWith -gt [version]$runnerHas) {
    $fix = if ($runnerVs) { "`"${runnerVs}`"" } else { "the Visual Studio version this runner reports" }
    Write-Warning ("Build MSVC version is newer than this runner's runtime, so " +
                   "binaries will fail to load. Set EIGEN_CI_MSVC_VS_VERSION in " +
                   "ci/build.windows.gitlab-ci.yml to ${fix}.")
  }
}
