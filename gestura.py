import cv2
import numpy as np
import mediapipe as mp
import dearpygui.dearpygui as dpg
import time

import engine 
from engine import GestureEngine

# ================== Global Variable ==================
engine_running = False 
cap = None 

# ================== Mediapipe setup ==================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cam_width = 640
cam_height = 480

def get_hand_points_mediapipe(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None, None

    hand_landmarks = result.multi_hand_landmarks[0]
    points = []
    for lm in hand_landmarks.landmark:
        x = lm.x * w
        y = lm.y * h
        points.append([x, y])

    return np.array(points, dtype=np.float32), hand_landmarks


def log_message(message, color=(148, 163, 184)):
    timestamp = time.strftime('%H:%M:%S')
    dpg.add_text(f"[{timestamp}] {message}", color=color, parent="log_group")
    y_scroll = dpg.get_y_scroll_max("log_window")
    dpg.set_y_scroll("log_window", y_scroll)


def engine_control(sender, app_data, user_data):
    global engine_running, cap, cam_width, cam_height
    
    if user_data == "START" and not engine_running:
        log_message("[SYSTEM] Initializing Hardware and Model...", color=(148, 163, 184))
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_message("[ERROR] Failed to open camera!", color=(248, 113, 113))
            return
            
        cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        engine_running = True
        
        dpg.set_value("status_text", "ACTIVE / RUNNING")
        dpg.configure_item("status_text", color=(74, 222, 128)) 
        log_message("[SUCCESS] Engine Started. Model Ready.", color=(26, 188, 156))

    elif user_data == "TERMINATE" and engine_running:
        log_message("[SYSTEM] Terminating processes...", color=(250, 204, 21))
        
        engine_running = False
        
        if cap is not None:
            cap.release()
            cap = None
            
        blank_texture = np.full((cam_height, cam_width, 4), 0.08, dtype=np.float32)
        dpg.set_value("camera_texture", blank_texture)
        
        dpg.set_value("status_text", "DISCONNECTED")
        dpg.configure_item("status_text", color=(248, 113, 113))
        log_message("[INFO] All systems released safely.", color=(148, 163, 184))


# ================== DPG SETUP ==================
dpg.create_context()

texture_data = np.full((cam_height, cam_width, 4), 0.08, dtype=np.float32)
with dpg.texture_registry(show=False):
    dpg.add_raw_texture(
        width=cam_width,
        height=cam_height,
        default_value=texture_data,
        format=dpg.mvFormat_Float_rgba,
        tag="camera_texture",
    )

with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (10, 20, 24), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (18, 30, 35), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Border, (35, 55, 60), category=dpg.mvThemeCat_Core)
        
        dpg.add_theme_color(dpg.mvThemeCol_Button, (22, 160, 133), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (26, 188, 156), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (18, 130, 108), category=dpg.mvThemeCat_Core)
        
        dpg.add_theme_color(dpg.mvThemeCol_Header, (30, 60, 65), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (40, 80, 85), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, (26, 188, 156), category=dpg.mvThemeCat_Core)
        
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (26, 188, 156), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_FrameBg, (10, 20, 24), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (25, 45, 50), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Text, (230, 240, 235), category=dpg.mvThemeCat_Core)
        
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 6)
        dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
        
        # PERBAIKAN PADDING: Dikecilkan lagi sedikit untuk ItemSpacing Y
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 10, 6) # Jarak vertikal dikurangi menjadi 6px
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 10, 10) 

with dpg.window(tag="PrimaryWindow", no_scrollbar=True, no_move=True, no_collapse=True, no_title_bar=True):
    
    with dpg.group(horizontal=True):
        
        # ================== 1. SIDEBAR KIRI ==================
        with dpg.child_window(width=280, border=False, no_scrollbar=True):
            
            with dpg.child_window(height=85, border=True, no_scrollbar=True):
                dpg.add_spacer(height=8)
                dpg.add_text("  GESTURA ENGINE", color=(26, 188, 156)) 
                dpg.add_spacer(height=2)
                dpg.add_text("  Admin Panel", color=(148, 163, 184))

            dpg.add_spacer(height=10)

            # Tinggi diturunkan sedikit dari 220px ke 210px
            with dpg.child_window(height=210, border=True, no_scrollbar=True):
                dpg.add_spacer(height=5)
                dpg.add_text("  MAIN MENU", color=(148, 163, 184))
                dpg.add_separator()
                dpg.add_spacer(height=8)
                dpg.add_button(label=" Dashboard Analisis", width=-1, height=35)
                dpg.add_spacer(height=2)
                dpg.add_button(label=" Start Tracking", width=-1, height=35, callback=engine_control, user_data="START")
                dpg.add_spacer(height=2)
                dpg.add_button(label=" Terminate Sistem", width=-1, height=35, callback=engine_control, user_data="TERMINATE")
                
            dpg.add_spacer(height=10)

            with dpg.child_window(height=-1, border=True, no_scrollbar=True):
                dpg.add_spacer(height=5)
                dpg.add_text("  KONFIGURASI", color=(148, 163, 184))
                dpg.add_separator()
                dpg.add_spacer(height=5)
                dpg.add_slider_int(label="K-Value", default_value=3, min_value=1, max_value=15, width=150)
                dpg.add_slider_float(label="Threshold", default_value=0.75, min_value=0.0, max_value=1.0, width=150)
                dpg.add_spacer(height=10)
                dpg.add_checkbox(label=" Tampilkan Nodes", default_value=True, tag="show_lm_cb")
                dpg.add_spacer(height=15)
                dpg.add_button(label=" Reset Kalibrasi", width=-1, height=35)

        # ================== 2. AREA KERJA UTAMA ==================
        with dpg.group():
            
            with dpg.child_window(height=45, border=True, width=-1, no_scrollbar=True):
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=5, height=25)
                    dpg.add_text("Sistem Pakar |", color=(148, 163, 184))
                    dpg.add_text("Diagnosis Gestur Hand-Landmark Metode KNN", color=(230, 240, 235))

            with dpg.group(horizontal=True):
                # Tinggi keempat kartu indikator dikurangi dari 90px ke 80px
                with dpg.child_window(width=240, height=80, border=True, no_scrollbar=True):
                    dpg.add_text(" STATUS KONEKSI", color=(248, 113, 113)) 
                    dpg.add_separator()
                    dpg.add_spacer(height=6)
                    dpg.add_text("DISCONNECTED", tag="status_text", color=(248, 113, 113))

                with dpg.child_window(width=230, height=80, border=True, no_scrollbar=True):
                    dpg.add_text(" DATA MODEL", color=(26, 188, 156)) 
                    dpg.add_separator()
                    dpg.add_spacer(height=6)
                    dpg.add_text("K-Nearest Neighbor")

                with dpg.child_window(width=230, height=80, border=True, no_scrollbar=True):
                    dpg.add_text(" CONFIDENCE RATE", color=(74, 222, 128)) 
                    dpg.add_separator()
                    dpg.add_spacer(height=6)
                    dpg.add_progress_bar(label="", default_value=0.85, overlay="Akurasi: 85%", width=-1, height=20)

                with dpg.child_window(width=240, height=80, border=True, no_scrollbar=True):
                    dpg.add_text(" ACTIVE NODES", color=(250, 204, 21)) 
                    dpg.add_separator()
                    dpg.add_spacer(height=6)
                    dpg.add_text("21 Titik Landmark")

            with dpg.group(horizontal=True):
                
                # PERBAIKAN: Container Kamera disesuaikan presisi agar pas tanpa sisa berlebih di bagian bawah
                with dpg.child_window(width=660, height=520, border=True, no_scrollbar=True):
                    with dpg.group(horizontal=True):
                        dpg.add_text(" VISUALISASI KAMERA", color=(148, 163, 184))
                        dpg.add_spacer(width=20)
                        dpg.add_text("| TEKAN 'C' UNTUK SIMPAN", color=(250, 204, 21))
                    dpg.add_separator()
                    dpg.add_spacer(height=5)
                    # Container gambar dibiarkan 640x480
                    with dpg.child_window(width=640, height=480, border=False, no_scrollbar=True): 
                        dpg.add_image("camera_texture")

                # PERBAIKAN: Penyesuaian container kanan ke 520px
                with dpg.child_window(width=-1, height=520, border=True, no_scrollbar=True):
                    
                    dpg.add_text(" SPATIAL MATRIX", color=(148, 163, 184))
                    dpg.add_separator()
                    dpg.add_spacer(height=2) # Spacer diperkecil
                    
                    # Tinggi baris tabel bawaan sedikit memakan ruang, kita kurangi CellPadding default
                    with dpg.theme() as table_theme:
                        with dpg.theme_component(dpg.mvTable):
                            dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 4, 3) # Memadatkan baris tabel
                            
                    with dpg.table(header_row=True, borders_innerH=True, borders_innerV=False, borders_outerV=False, row_background=True) as matrix_table:
                        dpg.add_table_column(label="ID")
                        dpg.add_table_column(label="X")
                        dpg.add_table_column(label="Y")
                        for i in range(5):
                            with dpg.table_row():
                                dpg.add_text(f"N-{i}", color=(148, 163, 184))
                                dpg.add_text("0.000", tag=f"t_x_{i}", color=(26, 188, 156))
                                dpg.add_text("0.000", tag=f"t_y_{i}", color=(26, 188, 156))
                    dpg.bind_item_theme(matrix_table, table_theme)
                    
                    dpg.add_spacer(height=10) # Jarak ke log diperpendek
                    
                    dpg.add_text(" TERMINAL OUTPUT", color=(148, 163, 184))
                    dpg.add_separator()
                    
                    with dpg.child_window(width=-1, height=-1, border=False, tag="log_window"):
                        with dpg.group(tag="log_group"):
                            dpg.add_text("[INFO] Workspace Siap...", color=(148, 163, 184))

dpg.bind_theme(global_theme)
dpg.create_viewport(title="Gestura Engine - Admin Dashboard", width=1280, height=800, resizable=False)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("PrimaryWindow", True)

# ================== CUSTOM RENDER LOOP ==================
capture_cooldown = 0

while dpg.is_dearpygui_running():
    if engine_running and cap is not None:
        ret, cap_frame = cap.read()
        if ret:
            cap_frame = cv2.flip(cap_frame, 1)
            points, hand_landmarks = get_hand_points_mediapipe(cap_frame)
            
            show_landmarks = dpg.get_value("show_lm_cb")
            
            if points is not None:
                if show_landmarks:
                    mp_draw.draw_landmarks(cap_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for i, (x, y) in enumerate(points):
                        cv2.circle(cap_frame, (int(x), int(y)), 4, (156, 188, 26), -1) 
                        cv2.putText(
                            cap_frame, str(i), (int(x) + 6, int(y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                        )

                for i in range(5):
                    lm = hand_landmarks.landmark[i]
                    dpg.set_value(f"t_x_{i}", f"{lm.x:.3f}")
                    dpg.set_value(f"t_y_{i}", f"{lm.y:.3f}")

                if dpg.is_key_down(67) and capture_cooldown == 0:
                    A = points.flatten().reshape(1, -1)
                    GestureEngine.define_conture(A)
                    capture_cooldown = 15 

            if capture_cooldown > 0:
                capture_cooldown -= 1

            rgba_frame = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2RGBA)
            if rgba_frame.shape[1] != cam_width or rgba_frame.shape[0] != cam_height:
                 rgba_frame = cv2.resize(rgba_frame, (cam_width, cam_height))
                 
            texture_data = rgba_frame.astype(np.float32) / 255.0
            dpg.set_value("camera_texture", texture_data.flatten()) 

    dpg.render_dearpygui_frame()

if cap is not None:
    cap.release()
hands.close()
dpg.destroy_context()