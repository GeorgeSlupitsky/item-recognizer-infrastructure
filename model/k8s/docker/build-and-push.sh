#!/bin/bash

IMAGE_NAME="ray-worker"
IMAGE_TAG="2.46.0-py310-aarch64"
REGISTRY="slupitskyi"

FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "🔨 Building custom Ray image with OpenCV dependencies..."
echo "Image: ${FULL_IMAGE_NAME}"

docker build -t ${FULL_IMAGE_NAME} .

if [ $? -eq 0 ]; then
    echo "✅ Image built successfully!"
    
    echo "🚀 Pushing image to registry..."
    docker push ${FULL_IMAGE_NAME}
    
    if [ $? -eq 0 ]; then
        echo "✅ Image pushed successfully!"
        echo "📝 Update your ray-cluster-values.yaml with:"
        echo "image:"
        echo "  repository: ${REGISTRY}/${IMAGE_NAME}"
        echo "  tag: \"${IMAGE_TAG}\""
    else
        echo "❌ Failed to push image"
        exit 1
    fi
else
    echo "❌ Failed to build image"
    exit 1
fi 