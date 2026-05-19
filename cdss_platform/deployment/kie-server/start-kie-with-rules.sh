#!/usr/bin/env bash
set -euo pipefail

KIE_USER="${KIE_SERVER_USER:-kieserver}"
KIE_PASSWORD="${KIE_SERVER_PASSWORD:-${KIE_SERVER_PWD:-kieserver1!}}"
KIE_CONTAINER="${KIE_CONTAINER_ID:-acs-clinical-rules}"
KIE_GROUP_ID="${KIE_KJAR_GROUP_ID:-com.inorder.clinical}"
KIE_ARTIFACT_ID="${KIE_KJAR_ARTIFACT_ID:-acs-rules}"
KIE_VERSION="${KIE_KJAR_VERSION:-1.0.0}"
KIE_URL="http://localhost:8080/kie-server/services/rest/server"

cd /opt/jboss/wildfly/bin

./start_kie-wb.sh &
server_pid="$!"

stop_server() {
  kill -TERM "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap stop_server TERM INT

for _ in $(seq 1 120); do
  if curl -fsS "${KIE_URL}/readycheck" >/dev/null 2>&1; then
    break
  fi

  if ! kill -0 "$server_pid" 2>/dev/null; then
    wait "$server_pid"
    exit $?
  fi

  sleep 2
done

if ! curl -fsS "${KIE_URL}/readycheck" >/dev/null 2>&1; then
  echo "KIE Server did not become ready in time" >&2
  stop_server
  exit 1
fi

container_response="$(mktemp)"
if curl -fsS -u "${KIE_USER}:${KIE_PASSWORD}" -H "Accept: application/json" "${KIE_URL}/containers/${KIE_CONTAINER}" -o "${container_response}" 2>/dev/null \
  && grep -q '"type"[[:space:]]*:[[:space:]]*"SUCCESS"' "${container_response}"; then
  echo "KIE container ${KIE_CONTAINER} is already deployed"
else
  rm -f "${container_response}"
  payload=$(cat <<JSON
{"container-id":"${KIE_CONTAINER}","release-id":{"group-id":"${KIE_GROUP_ID}","artifact-id":"${KIE_ARTIFACT_ID}","version":"${KIE_VERSION}"}}
JSON
)

  response_file="$(mktemp)"
  http_code="$(curl -sS -o "${response_file}" -w "%{http_code}" \
    -u "${KIE_USER}:${KIE_PASSWORD}" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -X PUT \
    "${KIE_URL}/containers/${KIE_CONTAINER}" \
    -d "${payload}")"

  if [ "${http_code}" != "200" ] && [ "${http_code}" != "201" ]; then
    echo "Failed to deploy KIE container ${KIE_CONTAINER}; HTTP ${http_code}" >&2
    cat "${response_file}" >&2
    rm -f "${response_file}"
    stop_server
    exit 1
  fi

  cat "${response_file}"
  rm -f "${response_file}"
fi
rm -f "${container_response}"

wait "$server_pid"
