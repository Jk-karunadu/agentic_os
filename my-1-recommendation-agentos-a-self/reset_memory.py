import sqlite3
import os

DB_PATH = "data/agentos.db"

def reset_memory():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='failures'")
    if cursor.fetchone():
        cursor.execute("DELETE FROM failures")
        conn.commit()
        print("Failure Logs (Reflection Database) have been successfully wiped.")

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='lessons'")
    if cursor.fetchone():
        cursor.execute("DELETE FROM lessons")
        conn.commit()
        print("Procedural Memory (Lessons) has been successfully wiped.")
        print("The agent is now at a blank slate. You can begin your demonstration!")
    else:
        print("No lessons table found. System is already at a blank slate.")
        
    conn.close()

if __name__ == "__main__":
    reset_memory()
