#!/usr/bin/env sh
set -eu

if [ "$#" -lt 1 ]; then
    echo "usage: $0 /path/to/sourccey-desktop [--with-voice-dependencies]" >&2
    exit 2
fi

desktop_path=$1
profile=${2:-}
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
package_source=$(CDPATH= cd -- "$script_dir/.." && pwd)
python_path="$desktop_path/modules/lerobot-vulcan/.venv/bin/python"
if [ ! -x "$python_path" ]; then
    echo "Sourccey Desktop Python runtime not found at $python_path" >&2
    exit 2
fi

requirement=$package_source
if [ "$profile" = "--with-voice-dependencies" ]; then
    requirement="$package_source[voice]"
fi
uv pip install --python "$python_path" -e "$requirement"
module_dir="$desktop_path/modules/sourccey-voice"
mkdir -p "$module_dir"
if [ ! -e "$module_dir/config.toml" ]; then
    cp "$package_source/config/default.toml" "$module_dir/config.toml"
fi
echo "Installed Sourccey Voice. Configure $module_dir/config.toml before starting the host."

