#!/bin/bash

set -e

if [ -z "$PROCESSOR" ]
then
    echo "Starting predictor"
    python predictor.py
else
    echo "Starting processor"
    python processor-scheduler.py
fi
