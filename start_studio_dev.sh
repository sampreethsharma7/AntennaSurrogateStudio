#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

export ANTENNA_STUDIO_BUILD_CHANNEL=development
export SNOWBUDDY_DEVELOPMENT_LOG=1

exec bash start_studio.sh
