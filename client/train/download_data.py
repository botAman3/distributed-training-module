import psycopg2
import os

def download_and_save_images():
    DB_NAME = "images"
    DB_USER = "instance_0"
    DB_PASSWORD = "password0"
    DB_HOST = "localhost" 
    DB_PORT = "5433"

    # Connect to the DB
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute("SELECT image_id, image_label, image_data FROM instance_0_data")

    rows = cursor.fetchall()
    base_folder = "train"
    os.makedirs(base_folder, exist_ok=True)
    
    for image_id, image_label, image_data in rows:
        label_folder = os.path.join(base_folder, str(image_label))
        os.makedirs(label_folder, exist_ok=True)

        image_path = os.path.join(label_folder, f"{image_id}.png")

        with open(image_path, "wb") as f:
            f.write(image_data)

    print("Download and save complete.")

    cursor.close()
    conn.close()

# Call the function to execute it
download_and_save_images()