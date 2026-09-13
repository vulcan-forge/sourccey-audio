#!/usr/bin/env sh
set -eu

if [ "$#" -lt 1 ]; then
    echo "usage: $0 /path/to/python [--remove-config /path/to/config-dir]" >&2
    exit 2
fi

python_path=$1
uv pip uninstall --python "$python_path" sourccey-voice
if [ "${2:-}" = "--remove-config" ]; then
    if [ "$#" -lt 3 ]; then
        echo "--remove-config requires an explicit configuration directory" >&2
        exit 2
    fi
    config_dir=$3
    case "$config_dir" in
        /|"$HOME"|"")
            echo "refusing to remove unsafe configuration path: $config_dir" >&2
            exit 2
            ;;
    esac
    rm -rf -- "$config_dir"
    echo "Removed package and configuration from $config_dir."
else
    echo "Removed the package. Configuration and model caches were preserved."
fi

