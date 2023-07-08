docker build -t kits .

docker run -it --gpus all -v /mnt/ssd4/Data/kits23-dataset:/kits23-dataset --ipc=host --rm -v $(pwd):/code kits bash
