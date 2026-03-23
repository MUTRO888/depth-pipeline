#!/usr/bin/env bash
# Mac 一键启动入口
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/scripts/mac/install_and_run_mac.sh"
