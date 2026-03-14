import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QStackedWidget, 
                             QFrame, QSizePolicy)
from PyQt6.QtGui import QIcon, QPixmap, QFont
from PyQt6.QtCore import Qt, QSize

class SignAIDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        # Menggunakan title bar standar OS
        self.setWindowTitle("SignAI - Hand Gesture Recognition")
        self.resize(1150, 750)
        
        # Font modern ala Windows 11 (Segoe UI Variable atau Segoe UI)
        self.modern_font = QFont("Segoe UI", 10)
        self.setFont(self.modern_font)

        # Widget Utama
        self.central_widget = QWidget()
        self.central_widget.setStyleSheet("background-color: #f3f4f6;") # Warna dasar abu-abu sangat muda (Windows 11 style)
        self.setCentralWidget(self.central_widget)
        
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # ---------------------------------------------------------
        # 1. SIDEBAR KECIL (ICON ONLY)
        # ---------------------------------------------------------
        self.sidebar_mini = QFrame()
        self.sidebar_mini.setFixedWidth(70)
        self.sidebar_mini.setStyleSheet("background-color: #1c1f26; border: none;")
        
        mini_layout = QVBoxLayout(self.sidebar_mini)
        mini_layout.setContentsMargins(10, 20, 10, 20)
        
        # Logo Mini
        self.logo_mini = QLabel()
        pix = QPixmap("assets/logo.png")
        if not pix.isNull():
            self.logo_mini.setPixmap(pix.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        mini_layout.addWidget(self.logo_mini, 0, Qt.AlignmentFlag.AlignCenter)
        
        mini_layout.addSpacing(30)
        
        # Navigasi Mini (Gunakan ikon kecil)
        self.btn_dash_mini = self.create_nav_btn("assets/dashboardsmall1.png", "", True)
        self.btn_cam_mini = self.create_nav_btn("assets/teacherssmall1.png", "", True)
        mini_layout.addWidget(self.btn_dash_mini)
        mini_layout.addWidget(self.btn_cam_mini)
        
        mini_layout.addStretch()
        
        self.btn_logout_mini = self.create_nav_btn("assets/signoutsmall1.png", "", True)
        mini_layout.addWidget(self.btn_logout_mini)

        # ---------------------------------------------------------
        # 2. SIDEBAR BESAR (EXPANDED) - Struktur Konsisten
        # ---------------------------------------------------------
        self.sidebar_full = QFrame()
        self.sidebar_full.setFixedWidth(230)
        self.sidebar_full.setStyleSheet("background-color: #1c1f26; border: none;")
        
        full_layout = QVBoxLayout(self.sidebar_full)
        full_layout.setContentsMargins(15, 20, 15, 20)

        # Header Logo & SignAI (Koreksi Jarak)
        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(15) # Jarak spesifik antara logo dan teks
        self.logo_full = QLabel()
        if not pix.isNull():
            self.logo_full.setPixmap(pix.scaled(32, 32, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        
        self.brand_text = QLabel("SignAI")
        self.brand_text.setStyleSheet("color: white; font-size: 20px; font-weight: 700; letter-spacing: 1px;")
        
        brand_layout.addWidget(self.logo_full)
        brand_layout.addWidget(self.brand_text)
        brand_layout.addStretch()
        full_layout.addLayout(brand_layout)
        
        full_layout.addSpacing(40)

        # Dashboard & Logout sekarang pakai ikon kecil + teks manual (Struktur Sama)
        self.btn_dash_full = self.create_nav_btn("assets/dashboardsmall1.png", "  Dashboard", False)
        self.btn_cam_full = self.create_nav_btn("assets/teacherssmall1.png", "  Handsign Camera", False)
        
        full_layout.addWidget(self.btn_dash_full)
        full_layout.addWidget(self.btn_cam_full)
        
        full_layout.addStretch()
        
        # Sign Out (Struktur Sama)
        self.btn_logout_full = self.create_nav_btn("assets/signoutsmall1.png", "  Sign Out", False)
        # Memberikan aksen merah tipis pada hover untuk Logout
        self.btn_logout_full.setStyleSheet("""
            QPushButton { color: #ebedef; border: none; padding: 12px; text-align: left; border-radius: 4px; font-size: 13px; }
            QPushButton:hover { background-color: #e84118; color: white; }
        """)
        full_layout.addWidget(self.btn_logout_full)

        # ---------------------------------------------------------
        # 3. KONTEN UTAMA
        # ---------------------------------------------------------
        self.main_content = QWidget()
        self.content_layout = QVBoxLayout(self.main_content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)

        # Sub-Header (White Bar)
        self.header_bar = QFrame()
        self.header_bar.setFixedHeight(60)
        self.header_bar.setStyleSheet("background-color: white; border-bottom: 1px solid #e5e7eb;")
        header_layout = QHBoxLayout(self.header_bar)
        
        self.toggle_btn = QPushButton("☰")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setStyleSheet("""
            QPushButton { border: none; font-size: 20px; color: #1c1f26; padding: 5px; background: transparent; }
            QPushButton:hover { background-color: #f3f4f6; border-radius: 5px; }
        """)
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        header_layout.addWidget(self.toggle_btn)
        
        header_layout.addSpacing(20)
        page_title = QLabel("Dashboard Pages")
        page_title.setStyleSheet("font-size: 14px; font-weight: 600; color: #374151;")
        header_layout.addWidget(page_title)
        
        header_layout.addStretch()

        # Stacked Widget (Konten)
        self.pages = QStackedWidget()
        self.page_dashboard = QFrame()
        self.page_dashboard.setStyleSheet("background-color: #f9fafb;")
        
        dash_layout = QVBoxLayout(self.page_dashboard)
        dash_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 1. Membuat Label untuk Gambar Utama
        self.home_img = QLabel()
        pix_welcome = QPixmap("assets/Business merger-amico.png") # Ganti dengan path gambar utama Anda (misal: welcome_img.png)

        if not pix_welcome.isNull():
            # Menggunakan ukuran sedang (misal lebar 250px)
            self.home_img.setPixmap(pix_welcome.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

        self.home_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.home_img.setStyleSheet("margin-bottom: 10px;") # Memberi jarak ke teks di bawahnya
        dash_layout.addWidget(self.home_img)

        # 2. Membuat Label Teks "Welcome to SignAI"
        welcome_lbl = QLabel("Welcome to SignAI")
        welcome_lbl.setStyleSheet("""
            font-size: 28px; 
            font-weight: bold; 
            color: #111827; 
            font-family: 'Segoe UI';
        """)
        welcome_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dash_layout.addWidget(welcome_lbl)

        # 3. Menambahkan sub-teks (opsional agar lebih manis)
        sub_welcome_lbl = QLabel("Sistem Pengenalan Bahasa Isyarat Tangan Berbasis Algoritma KNN")
        sub_welcome_lbl.setStyleSheet("font-size: 14px; color: #6b7280;")
        sub_welcome_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dash_layout.addWidget(sub_welcome_lbl)
        
        self.pages.addWidget(self.page_dashboard)

        self.content_layout.addWidget(self.header_bar)
        self.content_layout.addWidget(self.pages)

        # Susun Sidebar ke Layout Utama
        self.main_layout.addWidget(self.sidebar_mini)
        self.main_layout.addWidget(self.sidebar_full)
        self.main_layout.addWidget(self.main_content)

        # State awal: Sembunyikan mini sidebar
        self.sidebar_mini.hide()

    def create_nav_btn(self, icon_path, text="", is_mini=False):
        btn = QPushButton(text)
        btn.setIcon(QIcon(icon_path))
        btn.setIconSize(QSize(20, 20))
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        
        if is_mini:
            btn.setFixedSize(50, 50)
            btn.setStyleSheet("""
                QPushButton { border: none; border-radius: 5px; background: transparent; }
                QPushButton:hover { background-color: #2c313c; }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton { 
                    color: #ebedef; border: none; padding: 12px; 
                    text-align: left; border-radius: 4px; font-size: 13px; 
                    font-weight: 500;
                }
                QPushButton:hover { background-color: #2c313c; color: white; }
            """)
        return btn

    def toggle_sidebar(self):
        if self.sidebar_full.isVisible():
            self.sidebar_full.hide()
            self.sidebar_mini.show()
        else:
            self.sidebar_mini.hide()
            self.sidebar_full.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SignAIDashboard()
    window.show()
    sys.exit(app.exec())