#docker buildx build --platform linux/arm64 -t image_elon .
docker buildx build --platform linux/arm64 --build-arg NO_CPUINFO=True -t image_elon .


docker buildx build --platform linux/arm64 -t image_elon --load .
