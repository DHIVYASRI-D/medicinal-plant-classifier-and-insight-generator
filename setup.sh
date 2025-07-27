#!/bin/bash
mkdir -p /tmp/.streamlit
chmod -R 777 /tmp/.streamlit

mkdir -p /tmp/matplotlib_cache
chmod -R 777 /tmp/matplotlib_cache

export MPLCONFIGDIR=/tmp/matplotlib_cache
