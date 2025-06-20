import os
from PIL import Image
import numpy as np

def insert_image_data(connection):
    data = []
    try:
        base_path = "/home/navneet/grpc_collective_reduce/server/db/mnist/train"
        output_base = "/home/navneet/grpc_collective_reduce/server/db/processed_images"
        
        # 1. Verify input directory exists
        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"Input directory not found: {base_path}")
        
        # 2. Create output directory with verbose debugging
        print(f"\nAttempting to create output directory: {output_base}")
        print(f"Parent directory exists: {os.path.exists(os.path.dirname(output_base))}")
        print(f"Parent directory writable: {os.access(os.path.dirname(output_base), os.W_OK)}")
        
        try:
            os.makedirs(output_base, mode=0o755, exist_ok=True)
            print(f"Successfully created output directory: {output_base}")
            print(f"Directory exists: {os.path.exists(output_base)}")
            print(f"Directory writable: {os.access(output_base, os.W_OK)}")
        except Exception as e:
            raise RuntimeError(f"Failed to create output directory: {e}")

        # 3. Process class directories
        for class_dir in sorted(os.listdir(base_path)):
            class_path = os.path.join(base_path, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            # 4. Process instances
            for i in range(3):
                instance_path = os.path.join(output_base, f"instance_{i}")
                class_instance_path = os.path.join(instance_path, f"class_{class_dir}")
                
                # Debug folder creation
                print(f"\nCreating directory: {class_instance_path}")
                try:
                    os.makedirs(class_instance_path, mode=0o755, exist_ok=True)
                    print(f"Successfully created: {class_instance_path}")
                except Exception as e:
                    print(f"ERROR creating directory: {e}")
                    print(f"Current working directory: {os.getcwd()}")
                    continue
                
                # Process files
                files = sorted(os.listdir(class_path))
                for file_idx, file_name in enumerate(files[i::3]):
                    try:
                        file_path = os.path.join(class_path, file_name)
                        output_path = os.path.join(class_instance_path, file_name)
                        
                        # Debug file operations
                        print(f"\nProcessing file {file_idx}: {file_name}")
                        print(f"Input path exists: {os.path.exists(file_path)}")
                        
                        # Read and store data
                        with open(file_path, 'rb') as f:
                            img_data = f.read()
                            data.append((int(class_dir), i, img_data))
                        
                        # Save image
                        try:
                            img_array = np.frombuffer(img_data, dtype=np.uint8)
                            if len(img_array) == 784:  # MNIST 28x28
                                img_array = img_array.reshape(28, 28)
                            img = Image.fromarray(img_array)
                            img.save(output_path)
                            print(f"Saved image to: {output_path}")
                            print(f"Output exists: {os.path.exists(output_path)}")
                        except Exception as img_error:
                            print(f"Image save error: {img_error}")
                            
                    except Exception as file_error:
                        print(f"File processing error: {file_error}")
                        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return data