#!/bin/bash

# Check if epoch argument is provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <epoch> <unique_id>"
    exit 1
fi

epoch=$1  # Use the provided epoch
unique_id=$2  # Use the provided unique_id

# Create directories if they do not exist
mkdir -p ${unique_id}/left
mkdir -p ${unique_id}/right

# Capture a snapshot from camera 0
if libcamera-still --camera 0 --autofocus-on-capture --quality 95 --ev 2 -o ${unique_id}/left/${unique_id}_${epoch}_left.jpg; then
    echo "Snapshot taken from camera 0"
    
    # Capture a snapshot from camera 1
    if libcamera-still --camera 1 --autofocus-on-capture --quality 95 --ev 2 -o ${unique_id}/right/${unique_id}_${epoch}_right.jpg; then
        echo "Snapshot taken from camera 1"
    else
        echo "Failed to capture snapshot from camera 1"
    fi
else
    echo "Failed to capture snapshot from camera 0"
fi