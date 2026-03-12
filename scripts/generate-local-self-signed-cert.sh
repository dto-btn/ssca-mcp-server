#!/usr/bin/env bash
set -euo pipefail

# Generates localhost TLS assets for the orchestrator dev server.
#
# Prefers mkcert for trusted local certificates and falls back to OpenSSL
# self-signed output when mkcert is unavailable.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_DIR="${1:-$ROOT_DIR/certs}"
CERT_FILE="$CERT_DIR/localhost.crt"
KEY_FILE="$CERT_DIR/localhost.key"
VALID_DAYS="${ORCHESTRATOR_CERT_DAYS:-365}"

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required but was not found in PATH"
  exit 1
fi

mkdir -p "$CERT_DIR"

if command -v mkcert >/dev/null 2>&1; then
  echo "mkcert detected; generating locally trusted certificate"
  mkcert -install >/dev/null 2>&1 || true
  mkcert -cert-file "$CERT_FILE" -key-file "$KEY_FILE" localhost 127.0.0.1
  chmod 600 "$KEY_FILE"

  echo "Generated trusted local cert: $CERT_FILE"
  echo "Generated private key:       $KEY_FILE"
  echo "Use with:"
  echo "  ORCHESTRATOR_TLS_CERT_FILE=$CERT_FILE ORCHESTRATOR_TLS_KEY_FILE=$KEY_FILE ./scripts/start-playground-orchestrator.sh"
  exit 0
fi

TMP_CONFIG_FILE="$(mktemp)"
cleanup() {
  rm -f "$TMP_CONFIG_FILE"
}
trap cleanup EXIT

cat > "$TMP_CONFIG_FILE" <<EOF
[req]
default_bits = 2048
prompt = no
default_md = sha256
distinguished_name = dn
x509_extensions = v3_req

[dn]
C = CA
ST = ON
L = Ottawa
O = SSC Local Development
OU = Orchestrator MCP
CN = localhost

[v3_req]
basicConstraints = CA:FALSE
keyUsage = digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
EOF

openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout "$KEY_FILE" \
  -out "$CERT_FILE" \
  -days "$VALID_DAYS" \
  -config "$TMP_CONFIG_FILE"

chmod 600 "$KEY_FILE"

echo "Generated self-signed cert: $CERT_FILE"
echo "Generated private key:     $KEY_FILE"
echo "Warning: This cert is not automatically trusted by browsers."
echo "For seamless browser HTTPS, install mkcert and re-run this script."
echo "Use with:"
echo "  ORCHESTRATOR_TLS_CERT_FILE=$CERT_FILE ORCHESTRATOR_TLS_KEY_FILE=$KEY_FILE ./scripts/start-playground-orchestrator.sh"
