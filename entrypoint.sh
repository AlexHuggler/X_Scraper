#!/bin/bash

if [[ ! -f config.json ]]; then
  echo "Config file not found. Creating sample config..."
  python scrape.py --init-config
else
  echo "Config file found."
fi
echo 'alias scrape="python scrape.py"' >>~/.bashrc

exec "$@"
