#!/usr/bin/env bash
# Prepare a test host and record its specs for the paper's setup section.
# Supports RHEL/Fedora/SUSE (dnf/zypper) and Debian/Ubuntu (apt).
set -euo pipefail

if [[ $EUID -ne 0 ]]; then echo "run as root" >&2; exit 1; fi

if command -v dnf >/dev/null; then
  dnf install -y epel-release 2>/dev/null || true
  dnf install -y fio sysbench python3-pip
elif command -v zypper >/dev/null; then
  zypper --non-interactive install fio sysbench python3-pip
elif command -v apt-get >/dev/null; then
  apt-get update && apt-get install -y fio sysbench python3-pip
else
  echo "unknown package manager; install fio and sysbench manually" >&2
fi

out=${1:-host_info.txt}
{
  echo "# host info $(date -Is)"
  echo "## kernel";   uname -a
  echo "## os";       cat /etc/os-release 2>/dev/null | head -4
  echo "## cpu";      lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Socket|NUMA node\(s\)'
  echo "## memory";   free -h
  echo "## disks";    lsblk -d -o NAME,MODEL,SIZE,ROTA,TYPE
  echo "## io schedulers"
  for q in /sys/block/*/queue/scheduler; do echo "$q: $(cat "$q")"; done
  echo "## virtualization"; systemd-detect-virt 2>/dev/null || echo unknown
  echo "## tuned profile"; tuned-adm active 2>/dev/null || echo "tuned not installed"
} | tee "$out"

echo
echo "NOTE: disable tuned/other auto-tuners during experiments so they don't fight ostuner:"
echo "  systemctl stop tuned"
