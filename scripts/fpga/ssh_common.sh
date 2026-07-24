#!/usr/bin/env bash

# Shared SSH defaults for the FPGA AE scripts. Environment variables override
# every default below; set FPGA_PROXY_JUMP="" to connect without a jump host.
FPGA_USER="${FPGA_USER:-aevaluator}"
FPGA_HOST="${FPGA_HOST:-safari-fpga7.ethz.ch}"
FPGA_SSH_KEY="${FPGA_SSH_KEY:-$HOME/aevaluator1}"
FPGA_PROXY_USER="${FPGA_PROXY_USER:-aevaluator1}"
FPGA_PROXY_HOST="${FPGA_PROXY_HOST:-safari-proxy.ethz.ch}"
FPGA_PROXY_JUMP="${FPGA_PROXY_JUMP-${FPGA_PROXY_USER}@${FPGA_PROXY_HOST}}"
FPGA_PROXY_COMMAND="${FPGA_PROXY_COMMAND:-}"

fpga_require_ssh_key() {
  [[ -z "$FPGA_SSH_KEY" ]] && return 0
  if [[ ! -f "$FPGA_SSH_KEY" || ! -r "$FPGA_SSH_KEY" ]]; then
    printf 'fpga ssh: key not found or not readable: %s\n' "$FPGA_SSH_KEY" >&2
    printf '          Mount it there, or set FPGA_SSH_KEY to the private key path.\n' >&2
    exit 1
  fi
}

fpga_ssh_opts() {
  local connect_timeout="${1:-8}"
  local server_alive_count="${2:-8}"
  local proxy_cmd proxy_key_q proxy_jump_q

  fpga_require_ssh_key
  SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout="$connect_timeout" -o StrictHostKeyChecking=accept-new
            -o ServerAliveInterval=15 -o ServerAliveCountMax="$server_alive_count"
            -o PasswordAuthentication=no -o NumberOfPasswordPrompts=0)
  [[ -n "$FPGA_SSH_KEY" ]] && SSH_OPTS+=(-i "$FPGA_SSH_KEY" -o IdentitiesOnly=yes)
  if [[ -n "$FPGA_PROXY_COMMAND" ]]; then
    SSH_OPTS+=(-o "ProxyCommand=$FPGA_PROXY_COMMAND")
  elif [[ -n "$FPGA_PROXY_JUMP" ]]; then
    printf -v proxy_key_q '%q' "$FPGA_SSH_KEY"
    printf -v proxy_jump_q '%q' "$FPGA_PROXY_JUMP"
    proxy_cmd="ssh -i $proxy_key_q -o IdentitiesOnly=yes -o BatchMode=yes -o PasswordAuthentication=no -o NumberOfPasswordPrompts=0 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=$connect_timeout -W %h:%p $proxy_jump_q"
    SSH_OPTS+=(-o "ProxyCommand=$proxy_cmd")
  fi
}

fpga_ssh_cmd() {
  local cmd
  printf -v cmd '%q ' ssh "${SSH_OPTS[@]}"
  printf '%s' "${cmd% }"
}

fpga_ssh_wrapper() {
  local wrapper
  wrapper="$(mktemp "${TMPDIR:-/tmp}/fpga-ssh.XXXXXX")" || return 1
  {
    printf '#!/usr/bin/env bash\n'
    printf 'exec'
    printf ' %q' ssh "${SSH_OPTS[@]}"
    printf ' "$@"\n'
  } > "$wrapper" || { rm -f "$wrapper"; return 1; }
  chmod 700 "$wrapper" || { rm -f "$wrapper"; return 1; }
  printf '%s' "$wrapper"
}
