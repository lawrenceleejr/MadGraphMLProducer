#!/bin/bash
# Setup script for downloading UFO models for MadGraphMLProducer
#
# This script downloads the RPVMSSM_UFO model required for
# RPV gluino production and other RPV SUSY processes.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_DIR/models"

echo "MadGraphMLProducer - UFO Model Setup"
echo "======================================"
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Models directory: $MODELS_DIR"
echo ""

# Create models directory if it doesn't exist
mkdir -p "$MODELS_DIR"

# Function to download and extract a model
download_model() {
    local name=$1
    local url=$2
    local target_dir="$MODELS_DIR/$name"

    if [ -d "$target_dir" ]; then
        echo "Model $name already exists at $target_dir"
        echo "  To re-download, remove the directory first: rm -rf $target_dir"
        return 0
    fi

    echo "Downloading $name..."

    # Create temporary directory
    local temp_dir=$(mktemp -d)
    cd "$temp_dir"

    # Download
    if command -v wget &> /dev/null; then
        wget -q "$url" -O model.tar.gz || wget -q "$url" -O model.zip || {
            # Try git clone for GitHub repos
            if [[ "$url" == *"github.com"* ]]; then
                git clone --depth 1 "$url" model_repo
                mv model_repo/* "$target_dir" 2>/dev/null || mv model_repo "$target_dir"
                cd "$PROJECT_DIR"
                rm -rf "$temp_dir"
                echo "  Downloaded $name successfully (git clone)"
                return 0
            fi
            echo "  Error: Failed to download $name"
            cd "$PROJECT_DIR"
            rm -rf "$temp_dir"
            return 1
        }
    elif command -v curl &> /dev/null; then
        curl -sL "$url" -o model.tar.gz || curl -sL "$url" -o model.zip || {
            echo "  Error: Failed to download $name"
            cd "$PROJECT_DIR"
            rm -rf "$temp_dir"
            return 1
        }
    else
        echo "  Error: Neither wget nor curl found. Please install one."
        cd "$PROJECT_DIR"
        rm -rf "$temp_dir"
        return 1
    fi

    # Extract
    if [ -f model.tar.gz ]; then
        tar -xzf model.tar.gz
        rm model.tar.gz
    elif [ -f model.zip ]; then
        unzip -q model.zip
        rm model.zip
    fi

    # Move to target directory
    # Handle different archive structures
    local extracted_dir=$(ls -d */ 2>/dev/null | head -1)
    if [ -n "$extracted_dir" ]; then
        mv "$extracted_dir" "$target_dir"
    else
        mkdir -p "$target_dir"
        mv * "$target_dir/" 2>/dev/null || true
    fi

    cd "$PROJECT_DIR"
    rm -rf "$temp_dir"

    echo "  Downloaded $name successfully"
}

# Download RPVMSSM_UFO model
# This model includes RPV couplings for gluino -> 3 jets decay
echo ""
echo "Downloading RPVMSSM_UFO model..."
echo "  Source: https://github.com/ilmonteux/RPVMSSM_UFO"
echo ""

download_model "RPVMSSM_UFO" "https://github.com/ilmonteux/RPVMSSM_UFO"

# Verify the model
if [ -d "$MODELS_DIR/RPVMSSM_UFO" ]; then
    if [ -f "$MODELS_DIR/RPVMSSM_UFO/__init__.py" ] || [ -f "$MODELS_DIR/RPVMSSM_UFO/particles.py" ]; then
        echo ""
        echo "Model verification: RPVMSSM_UFO looks valid"
    else
        echo ""
        echo "Warning: RPVMSSM_UFO directory exists but may be incomplete"
        echo "  Please check: $MODELS_DIR/RPVMSSM_UFO"
    fi
fi

echo ""
echo "======================================"
echo "Setup complete!"
echo ""
echo "Available models:"
for dir in "$MODELS_DIR"/*/; do
    if [ -d "$dir" ]; then
        echo "  - $(basename "$dir")"
    fi
done
echo ""
echo "To use a model in MadGraph, specify it in your configuration:"
echo "  process:"
echo "    model: \"RPVMSSM_UFO\""
echo ""
