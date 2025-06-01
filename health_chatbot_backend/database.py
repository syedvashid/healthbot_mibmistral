import mysql.connector
from dotenv import load_dotenv
import os

# Load variables from .env
load_dotenv()

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("DB_HOST", "127.0.0.1"),  # Use 127.0.0.1 for TCP/IP
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_DATABASE"),
            port=3306
        )
        print("Database connection successful!")
        return conn
    except mysql.connector.Error as e:
        print(f"Database connection failed: {e}")
        return None
