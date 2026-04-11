import sqlite3
import os



class DatabaseManager:
    def __init__(self, db_path="gestura.db"):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()            
            columns = ", ".join([f"point_{i}x REAL, point_{i}y REAL" for i in range(21)])
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS hand_dataset (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL,
                    {columns}
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    predicted_char TEXT,
                    confidence REAL,
                    latency_ms REAL
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            conn.commit()

    def add_hand_data(self, label, coordinates):
        placeholders = ", ".join(["?"] * 43) 
        query = f"INSERT INTO hand_dataset (label, {self._get_coord_column_names()}) VALUES ({placeholders})"
        
        with self.get_connection() as conn:
            conn.execute(query, [label] + list(coordinates))
            conn.commit()

    def fetch_all_training_data(self):
        # """Mengambil semua data untuk di-fit ke model KNN di engine.py."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM hand_dataset")
            return cursor.fetchall()

    def log_inference(self, char, conf, latency):
        # """Mencatat hasil prediksi ke database."""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO inference_logs (predicted_char, confidence, latency_ms)
                VALUES (?, ?, ?)
            ''', (char, conf, latency))
            conn.commit()

    def get_recent_logs(self, limit=10):
        # """Mengambil log terbaru untuk ditampilkan di UI Log."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, predicted_char, confidence FROM inference_logs ORDER BY id DESC LIMIT ?", (limit,))
            return cursor.fetchall()

    def _get_coord_column_names(self):
        # """Helper untuk mendapatkan nama kolom koordinat."""
        names = []
        for i in range(21):
            names.extend([f"point_{i}x", f"point_{i}y"])
        return ", ".join(names)

db = DatabaseManager("gestura.db")


# -------------------
# Database testing code
# -------------------
if __name__ == "__main__":
    print("Database initialized and ready to use.")
    
    try : 
        print("Testing database connection and table creation...")
        db.log_inference("A", 0.95, 12.5)
        print("Inference log added successfully.")
        
        print("Fetching recent logs...")
        logs = db.get_recent_logs(limit=2)
        
        if logs:
            print("Recent Logs:")
            print(f"{logs}")
            print("Database test completed successfully.")
        else: 
            print("No logs found. Database test failed.")
            
        
    except Exception as e:
        print(f"Database test failed with error: {e}")
        print(f"Detail error: {e}")    