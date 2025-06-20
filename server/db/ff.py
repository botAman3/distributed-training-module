import os
folder_path =  "/home/navneet/grpc_collective_reduce/server/db/mnist_jpeg/train/0"

total_items = len(os.listdir(folder_path))
print(f"Total items in folder: {total_items}")