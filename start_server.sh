#!/bin/bash
cd "$(dirname "$0")"
source openvino-env/bin/activate
python server_production.py
