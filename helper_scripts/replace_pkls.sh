#!/usr/bin/env bash
set -euo pipefail

# Run this script from the repository root that contains pkl_* dirs and the models/ directory.
# Usage:
#   helper_scripts/replace_pkls.sh --models MODEL [MODEL ...] [--sources ARTIFACT_ZIP ...]
# Where each ARTIFACT_ZIP is a GitLab CI artifacts zip that contains
#   models/pkl_*.tar.gz sub-archives. The script will extract available
#   pkl_* sub-archives and copy the relevant .pkl files for the given
#   models into models/<model>/ref/.
#
# Arguments format:
#   --models  followed by one or more model names
#   --sources followed by zero or more artifact zip paths

print_usage() {
  echo "Usage: $(basename "$0") --models MODEL [MODEL ...] [--sources ARTIFACT_ZIP ...]" >&2
  echo "Examples:" >&2
  echo "  $(basename "$0") --models 2ph_comp 3ph_do phreeqc_dissolution" >&2
  echo "  $(basename "$0") --models 2ph_comp --sources 'artifacts - 2025-09-15T173748.573.zip'" >&2
  echo "  $(basename "$0") --models 2ph_comp 3ph_do --sources 'artifacts - 2025-09-15T173748.573.zip' 'artifacts - 2025-09-15T173814.785.zip'" >&2
}

if [[ $# -lt 1 ]]; then
  print_usage
  exit 1
fi

models=()
artifact_zips=()

parsing_models=false
parsing_sources=false

for arg in "$@"; do
  case "$arg" in
    --models)
      parsing_models=true
      parsing_sources=false
      ;;
    --sources)
      parsing_sources=true
      parsing_models=false
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      if $parsing_models; then
        models+=("$arg")
      elif $parsing_sources; then
        artifact_zips+=("$arg")
      else
        echo "ERROR: Unexpected argument '$arg'. Use --models and optionally --sources." >&2
        print_usage
        exit 1
      fi
      ;;
  esac
done

if [[ ${#models[@]} -eq 0 ]]; then
  echo "ERROR: No models specified." >&2
  print_usage
  exit 1
fi

models_dir="./models"
if [[ ! -d "$models_dir" ]]; then
  echo "ERROR: '$models_dir' directory not found. Run this script from the repo root." >&2
  exit 1
fi

# Copy order
src_dirs=( "pkl_lin" "pkl_lin_odls" "pkl_win_odls" "pkl_win" "pkl_gpu" )

# Map source dir -> expected filename
declare -A src_file=(
  ["pkl_lin"]="perf_lin_iter.pkl"
  ["pkl_lin_odls"]="perf_lin_odls.pkl"
  ["pkl_win_odls"]="perf_win_odls.pkl"
  ["pkl_win"]="perf_win_iter.pkl"
  ["pkl_gpu"]="perf_lin_gpu.pkl"
)

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: Required command '$cmd' not found in PATH" >&2
    exit 1
  fi
}

# Copy from a given root that contains pkl_* directories
copy_from_root() {
  local root_dir="$1"
  for model in "${models[@]}"; do
    local dest="${models_dir}/${model}/ref"
    mkdir -p "$dest"

    for subdir in "${src_dirs[@]}"; do
      local copied=false
      # Prefer direct path first
      local direct_dir="${root_dir}/${subdir}"
      if [[ -d "$direct_dir" ]]; then
        local src="${direct_dir}/${model}/ref/${src_file[$subdir]}"
        if [[ -f "$src" ]]; then
          cp -v "$src" "$dest"/
          copied=true
        fi
      fi

      # If not found at direct path, search for any nested directories named $subdir
      if [[ "$copied" == false ]]; then
        mapfile -t candidate_dirs < <(find "$root_dir" -type d -name "$subdir" -print 2>/dev/null || true)
        for cdir in "${candidate_dirs[@]}"; do
          local src2="${cdir}/${model}/ref/${src_file[$subdir]}"
          if [[ -f "$src2" ]]; then
            cp -v "$src2" "$dest"/
            copied=true
            # don't break; allow later zips/specs to overwrite with newer files
          fi
        done
      fi
      if [[ "$copied" == false ]]; then
        echo "WARN: missing ${subdir}/${model}/ref/${src_file[$subdir]} under $root_dir" >&2
      fi
    done
  done
}

# If artifact zips are provided, extract and copy from them; otherwise copy from local pkl_* dirs
if [[ ${#artifact_zips[@]} -gt 0 ]]; then
  require_cmd unzip
  require_cmd tar

  # ensure cleanup of temp dirs
  cleanup_dirs=()
  trap 'if [[ ${#cleanup_dirs[@]} -gt 0 ]]; then rm -rf "${cleanup_dirs[@]}"; fi' EXIT

  for zip_path in "${artifact_zips[@]}"; do
    tmp_dir=$(mktemp -d -t pkls_zip_XXXXXXXX)
    cleanup_dirs+=("$tmp_dir")
    unzip -q "$zip_path" -d "$tmp_dir"

    # Find pkl_*.tar.gz or pkl_*.zip sub-archives anywhere under extracted content (commonly under */models/)
    mapfile -t sub_archives < <(find "$tmp_dir" -type f \( -name 'pkl_*.tar.gz' -o -name 'pkl_*.zip' \) -print 2>/dev/null || true)
    if [[ ${#sub_archives[@]} -eq 0 ]]; then
      echo "WARN: No pkl_*.tar.gz sub-archives found in $zip_path" >&2
      continue
    fi

    for sub in "${sub_archives[@]}"; do
      base=$(basename "$sub")
      spec="$base"
      spec="${spec%.tar.gz}"
      spec="${spec%.zip}"
      extract_dir=$(mktemp -d -t "${spec}_XXXXXXXX")
      cleanup_dirs+=("$extract_dir")
      if [[ "$sub" == *.tar.gz ]]; then
        tar -xzf "$sub" -C "$extract_dir"
      else
        unzip -q "$sub" -d "$extract_dir"
      fi

      # Look for directories named exactly as each model inside the extracted tar
      for model in "${models[@]}"; do
        dest="${models_dir}/${model}/ref"
        mkdir -p "$dest"
        found_any=false

        # Find any directory matching the model name anywhere under the extracted content
        while IFS= read -r -d '' model_dir; do
          # Prefer .pkl files directly under the model directory
          mapfile -d '' model_pkls < <(find "$model_dir" -maxdepth 1 -type f -name '*.pkl' -print0 2>/dev/null || true)
          if [[ ${#model_pkls[@]} -eq 0 && -d "$model_dir/ref" ]]; then
            # Fallback: look under a nested ref directory if present
            mapfile -d '' model_pkls < <(find "$model_dir/ref" -maxdepth 1 -type f -name '*.pkl' -print0 2>/dev/null || true)
          fi
          if [[ ${#model_pkls[@]} -gt 0 ]]; then
            cp -v -- "${model_pkls[@]}" "$dest"/
            found_any=true
          fi
        done < <(find "$extract_dir" -type d -name "$model" -print0 2>/dev/null || true)

        if [[ "$found_any" == false ]]; then
          echo "WARN: No .pkl files for model '$model' found in $base" >&2
        fi
      done
    done
  done
else
  # Fallback: copy from local pkl_* directories in the repo root
  copy_from_root "."
fi
