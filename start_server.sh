#!/bin/bash
cd "$(dirname "$0")"
source npglue-env/bin/activate
python server_production.py
