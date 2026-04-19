import sqlite3
from datetime import datetime
class DatabaseManager:
    def __init__(self, db_path="gestura.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

    def get_connection(self):
        """Membuka koneksi baru untuk transaksi terisolasi."""
        return sqlite3.connect(self.db_path)

    def add_hand_data(self, label, coordinates):
        """Menambahkan data landmark tangan ke dataset."""
        placeholders = ", ".join(["?"] * 43) 
        query = f"INSERT INTO hand_dataset (label, {self._get_coord_column_names()}) VALUES ({placeholders})"
        
        with self.get_connection() as conn:
            conn.execute(query, [label] + list(coordinates))
            conn.commit()

    def fetch_all_training_data(self):
        """Mengambil semua data untuk di-fit ke model KNN di engine.py."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM hand_dataset")
            return cursor.fetchall()

    def log_inference(self, char, conf, latency):
        """Mencatat hasil prediksi ke database."""
        with self.get_connection() as conn:
            conn.execute('''
                INSERT INTO inference_logs (predicted_char, confidence, latency_ms)
                VALUES (?, ?, ?)
            ''', (char, conf, latency))
            conn.commit()

    def get_recent_logs(self, limit=10):
        """Mengambil log terbaru untuk ditampilkan di UI Log."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, predicted_char, confidence FROM inference_logs ORDER BY id DESC LIMIT ?", (limit,))
            return cursor.fetchall()

    def _get_coord_column_names(self):
        """Helper untuk mendapatkan nama kolom koordinat."""
        names = []
        for i in range(21):
            names.extend([f"point_{i}x", f"point_{i}y"])
        return ", ".join(names)
    
    def auth_user(self, username):
        """Mengambil password untuk autentikasi user aktif."""
        # Sangat disarankan menambahkan pengecekan isactive = 1
        # agar user yang sudah di-nonaktifkan tidak bisa login
        cursor = self.conn.execute(
            "SELECT password FROM msuser WHERE username = ? AND isactive = 1", 
            (username,)
        )
        return cursor.fetchone()
    
    def create_user(self, username, password_hash):
        """Membuat user baru dengan data audit yang terisi otomatis."""
        # Generate waktu saat ini (contoh: 2026-04-20 14:30:00)
        waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Default value untuk kolom tambahan
        isactive = 1          # 1 = Aktif, 0 = Non-aktif
        created_by = username # Mencatat siapa yang membuat (bisa juga diisi "system")

        try:
            query = """
                INSERT INTO msuser 
                (username, password, createdby, updatedby, createddate, updateddate, isactive) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            # Sesuaikan urutan value dengan tanda tanya (?) di atas
            values = (username, password_hash, created_by, created_by, waktu_sekarang, waktu_sekarang, isactive)
            
            self.conn.execute(query, values)
            self.conn.commit()
            return True
            
        except sqlite3.IntegrityError as e:
            # Opsional: Print error ke terminal untuk memudahkan debugging di masa depan
            print(f"[DEBUG] Gagal insert ke database: {e}")
            return False