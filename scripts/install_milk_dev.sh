#!/usr/bin/env bash
# ==============================================================================
# install_milk_dev.sh - Automated installer for Milk framework (framework-dev)
# ==============================================================================
# Clones, compiles, and installs the Milk framework required for GRIC's FPS
# streaming daemon (milk-fpsexec-gric-cluster) and Milk CLI module (milkgric).
# ==============================================================================
set -euo pipefail

PREFIX="/usr/local"
BRANCH="framework-dev"
SRC_DIR="${HOME}/src/milk"
INSTALL_DEPS=false
USE_SUDO=true

print_usage()
{
    cat << 'EOF'
Usage: install_milk_dev.sh [OPTIONS]

Automated installer for the Milk framework (framework-dev branch).

Options:
  -p, --prefix <DIR>     Installation prefix (default: /usr/local)
  -b, --branch <BRANCH>  Git branch to clone (default: framework-dev)
  -s, --src-dir <DIR>    Directory to clone Milk source (default: ~/src/milk)
      --no-sudo          Do not use sudo for installation / ldconfig
  -d, --deps             Install required Debian/Ubuntu apt packages
  -h, --help             Display this help message and exit

Examples:
  ./scripts/install_milk_dev.sh --deps
  ./scripts/install_milk_dev.sh --prefix /usr/local/milk
  ./scripts/install_milk_dev.sh --prefix ~/.local --no-sudo
EOF
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--prefix)
            PREFIX="$2"
            shift 2
            ;;
        -b|--branch)
            BRANCH="$2"
            shift 2
            ;;
        -s|--src-dir)
            SRC_DIR="$2"
            shift 2
            ;;
        --no-sudo)
            USE_SUDO=false
            shift
            ;;
        -d|--deps)
            INSTALL_DEPS=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

# Check if sudo is needed
if [ "$USE_SUDO" = true ] && [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
else
    SUDO=""
fi

if [ "$INSTALL_DEPS" = true ]; then
    echo "==> Installing system dependencies..."
    $SUDO apt-get update
    $SUDO apt-get install -y \
        build-essential cmake pkg-config git \
        libcfitsio-dev libreadline-dev libncurses-dev \
        libgsl-dev libomp-dev libfftw3-dev bison flex
fi

echo "==> Preparing Milk source in: ${SRC_DIR}"
mkdir -p "$(dirname "${SRC_DIR}")"

if [ -d "${SRC_DIR}/.git" ]; then
    echo "==> Existing repository found. Updating ${BRANCH}..."
    git -C "${SRC_DIR}" fetch origin
    git -C "${SRC_DIR}" checkout "${BRANCH}"
    git -C "${SRC_DIR}" pull --ff-only origin "${BRANCH}" || true
    git -C "${SRC_DIR}" submodule update --init --recursive
else
    echo "==> Cloning Milk (${BRANCH}) recursively..."
    git clone --recursive -b "${BRANCH}" \
        https://github.com/milk-org/milk.git "${SRC_DIR}"
fi

echo "==> Configuring Milk (Prefix: ${PREFIX})..."
cmake -B "${SRC_DIR}/_build" -S "${SRC_DIR}" \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF \
    -DINSTALLMAKEDEFAULT=ON

NPROC="$(nproc 2>/dev/null || echo 4)"
echo "==> Building Milk with ${NPROC} parallel jobs..."
cmake --build "${SRC_DIR}/_build" -j"${NPROC}"

echo "==> Installing Milk to ${PREFIX}..."
if [ -n "$SUDO" ] && [ ! -w "${PREFIX}" ]; then
    $SUDO cmake --install "${SRC_DIR}/_build"
    if command -v ldconfig &>/dev/null; then
        $SUDO ldconfig || true
    fi
else
    cmake --install "${SRC_DIR}/_build"
fi

echo ""
echo "=================================================================="
echo " Milk Framework installation complete!"
echo "=================================================================="
echo ""

PKG_MATCH=false
PKG_DIR=""
MILK_PC="$(find "${PREFIX}" -name "milk.pc" 2>/dev/null | head -n 1 || true)"
if [ -n "${MILK_PC}" ]; then
    PKG_MATCH=true
    PKG_DIR="$(dirname "${MILK_PC}")"
    ACTUAL_ROOT="$(cd "${PKG_DIR}/../.." && pwd)"
    if [ ! -e "/usr/local/milk" ]; then
        if [ -n "$SUDO" ]; then
            $SUDO ln -snf "${ACTUAL_ROOT}" /usr/local/milk 2>/dev/null || true
        else
            ln -snf "${ACTUAL_ROOT}" /usr/local/milk 2>/dev/null || true
        fi
    fi
fi

if [ "$PKG_MATCH" = true ]; then
    echo "Found milk.pc in: ${PKG_DIR}"
    if ! pkg-config --exists milk 2>/dev/null; then
        echo ""
        echo "Notice: ${PKG_DIR} is not currently in PKG_CONFIG_PATH."
        echo "Add the following lines to your ~/.bashrc:"
        echo "  export PKG_CONFIG_PATH=\"${PKG_DIR}:\$PKG_CONFIG_PATH\""
        echo "  export PATH=\"${PREFIX}/bin:\$PATH\""
        echo "  export LD_LIBRARY_PATH=\"${PREFIX}/lib:\$LD_LIBRARY_PATH\""
        echo ""
    else
        echo "Verified: pkg-config located milk $(pkg-config --modversion milk)"
        echo "Verified: pkg-config located ImageStreamIO $(pkg-config --modversion ImageStreamIO)"
    fi
fi

echo ""
echo "Next step: configure and build gric-cluster with Milk enabled:"
echo "  cd /path/to/gric-cluster"
echo "  mkdir -p build && cd build"
echo "  cmake .."
echo "  make -j\$(nproc)"
echo "  sudo make install"
echo ""
