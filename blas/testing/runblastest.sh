#!/bin/bash
# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

# Usage: runblastest.sh <driver> [<input.dat>]. The netlib drivers exit 0 even when a test fails, so the
# verdict comes from their reports: xBLAT1's stdout, and the <driver>.summ file xBLAT2/3 write.

driver=$1
data=$2
out="${driver}.out"
summ="${driver}.summ"

fail() {
  echo "Test ${driver} failed: $1"
  for f in "${out}" "${summ}"; do
    if [ -s "$f" ]; then
      echo "----- $f -----"
      cat "$f"
    fi
  done
  exit 1
}

rm -f "${out}" "${summ}" "${driver}.snap"
if [ -n "${data}" ] && [ -f "${data}" ]; then
  ./"${driver}" < "${data}" > "${out}" 2>&1 || fail "exit status $?"
  [ -f "${summ}" ] || fail "no summary file ${summ}"
  if grep -E -q "FATAL ERROR|FAILED|SUSPECT" "${summ}"; then
    fail "the summary reports a failure"
  fi
  grep -q "PASSED THE COMPUTATIONAL TESTS" "${summ}" || fail "no routine passed"
else
  ./"${driver}" < /dev/null > "${out}" 2>&1 || fail "exit status $?"
  if grep -w -q "FAIL" "${out}"; then
    fail "a case reports FAIL"
  fi
  grep -q "PASS" "${out}" || fail "no case passed"
fi
echo "Test ${driver} passed"
