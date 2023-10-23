#!/bin/bash

PUSHGATEWAY_HOSTNAME=${1}
PUSHGATEWAY_USER=${2}
PUSHGATEWAY_PASSWORD=${3}
PUSHGATEWAY_ADDITIONAL_PATH=/pushgateway

PROM_ADDITIONAL_LABELS=/service/ksmt
SLEEP_TIME_SECONDS=15
VERSION_CADVISOR=v0.36.0
VERSION_CURL=7.84.0
VERSION_NODE_EXPORTER=v1.3.1
PORT_CADVISOR=9280
PORT_NODE_EXPORTER=9100

# base linux system metrics
if ! netstat -tulpn | grep -q ${PORT_NODE_EXPORTER} ; then
  docker run -d --name node_exporter \
                --net="host" \
                --pid="host" \
                --volume="/:/host:ro,rslave" \
                    quay.io/prometheus/node-exporter:${VERSION_NODE_EXPORTER} \
                    --path.rootfs=/host
  docker run -d --name curl-node \
                --net="host" \
                --entrypoint=/bin/sh \
                    curlimages/curl:${VERSION_CURL} \
                  "-c" "while true; do curl localhost:9100/metrics | curl -u ${PUSHGATEWAY_USER}:${PUSHGATEWAY_PASSWORD} --data-binary @- https://${PUSHGATEWAY_HOSTNAME}${PUSHGATEWAY_ADDITIONAL_PATH}/metrics/job/pushgateway/instance/${GITHUB_RUN_ID}-${HOSTNAME}${PROM_ADDITIONAL_LABELS} ; sleep ${SLEEP_TIME_SECONDS}; done"
fi
