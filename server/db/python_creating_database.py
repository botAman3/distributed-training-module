import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from psycopg2.extras import execute_batch
import os
import csv
def create_connection():
    try:
        connection = psycopg2.connect(
            dbname="images",
            user="navneetk",
            password="1234@abcd",
            host="localhost",
            port="5433"
        )
        print("Connection to PostgreSQL database successful")
        return connection
    except Exception as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None


def create_table(connection):
    try:
        with connection.cursor() as cursor:
            delete_table_query = """
            DROP TABLE IF EXISTS images_data;
            """
            create_table_query = """
            CREATE TABLE IF NOT EXISTS images_data (
                        image_id SERIAL PRIMARY KEY,
                        image_label INTEGER NOT NULL,
                        instance_id INTEGER NOT NULL,
                        image_data BYTEA NOT NULL
                    );
            """
            cursor.execute(delete_table_query)
            cursor.execute(create_table_query)
            connection.commit()
            print("Table 'images' created successfully")
    except Exception as e:
                print(f"Error creating table: {e}")

def insert_image_data(connection):
    data = []
    try:
     base_path = "/home/navneet/grpc_collective_reduce/server/db/mnist/train"
     total_files = 0
     for class_dir in os.listdir(base_path):
        folder_path = os.path.join(base_path, class_dir)
        if os.path.isdir(folder_path):
            total_files += len(os.listdir(folder_path))
     print(f"Total files available: {total_files}")

     for class_dir in sorted(os.listdir(base_path)):
      folder_path = os.path.join(base_path, class_dir)
      if not os.path.isdir(folder_path):
        continue
    
      files_in_class = sorted(os.listdir(folder_path))
      for i in range(3):
        for file_name in files_in_class[i::3]:
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'rb') as file:
                    image_data = file.read()
                    image_label = int(class_dir)
                    data.append((image_label, i, image_data))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

     print(f"Total files processed: {len(data)}")
     output_csv = "/home/navneet/grpc_collective_reduce/server/db/mnist_labels.csv"
     with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for item in data:
             writer.writerow([item[0], item[1]])
             
     cur = connection.cursor()  
     insert_query = """
        INSERT INTO images_data (image_label, instance_id, image_data)
        VALUES (%s, %s, %s)
        """
        
     execute_batch(cur, insert_query, data, page_size=100)
     conn.commit()
     print(f"Successfully inserted {len(data)} records")
    
     print("Image data inserted successfully")
    except Exception as e:
        print(f"Error inserting image data: {e}")
                
if __name__ == "__main__":
    conn = create_connection()
    if conn:
        create_table(conn)
        insert_image_data(conn)
        conn.close()
    else:
        print("Failed to create database connection")



        