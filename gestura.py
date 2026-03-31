import cv2
import numpy as np
import mediapipe as mp
import dearpygui.dearpygui as dpg
import time

# ================== STATE MANAGEMENT ==================
# Variabel global untuk mengontrol apakah sistem "hidup" atau "mati"
engine_running = False 
cap = None # Kamera tidak diinisialisasi secara default

# ================== MEDIAPIPE SETUP ==================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# Set resolusi default untuk tekstur (bisa disesuaikan dengan kamera Anda)
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


def log_message(message, color=(200, 200, 200)):
    """Fungsi helper untuk menambah log ke UI"""
    timestamp = time.strftime('%H:%M:%S')
    dpg.add_text(f"[{timestamp}] {message}", color=color, parent="log_group")
    # Auto-scroll opsional jika log sudah terlalu penuh
    y_scroll = dpg.get_y_scroll_max("log_window")
    dpg.set_y_scroll("log_window", y_scroll)


def define_conture(A):
    # Menggunakan fungsi helper log
    log_message(f"[ACTION] Data Captured! Matrix shape: {A.shape}", color=(255, 255, 100))


def engine_control(sender, app_data, user_data):
    global engine_running, cap, cam_width, cam_height
    
    if user_data == "START" and not engine_running:
        log_message("[SYSTEM] Initializing Hardware and Model...", color=(150, 150, 150))
        
        # 1. Buka Kamera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_message("[ERROR] Failed to open camera!", color=(255, 0, 0))
            return
            
        # Update dimensi tekstur sesuai kamera yang berhasil dibuka
        cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 2. Set State
        engine_running = True
        
        # 3. Update UI
        dpg.set_value("status_text", "ACTIVE")
        dpg.configure_item("status_text", color=(0, 255, 0))
        log_message("[SUCCESS] Engine Started. Model Ready.", color=(100, 255, 150))

    elif user_data == "TERMINATE" and engine_running:
        log_message("[SYSTEM] Terminating processes...", color=(255, 150, 50))
        
        # 1. Stop State
        engine_running = False
        
        # 2. Release Hardware
        if cap is not None:
            cap.release()
            cap = None
            
        # 3. Kosongkan Tekstur (Jadikan Hitam)
        blank_texture = np.zeros((cam_height, cam_width, 4), dtype=np.float32)
        dpg.set_value("camera_texture", blank_texture)
        
        # 4. Update UI
        dpg.set_value("status_text", "DISCONNECTED")
        dpg.configure_item("status_text", color=(255, 0, 0))
        log_message("[INFO] All systems released safely.", color=(150, 150, 150))


# ================== DPG SETUP ==================
dpg.create_context()

texture_data = np.zeros((cam_height, cam_width, 4), dtype=np.float32)
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
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (15, 15, 15), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (24, 24, 28), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Border, (40, 40, 45), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Button, (45, 45, 55), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (100, 255, 150, 150), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Header, (40, 40, 50), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (100, 255, 150), category=dpg.mvThemeCat_Core)
        
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 10)

with dpg.window(tag="PrimaryWindow"):
    with dpg.group(horizontal=True):
        dpg.add_text(" GESTURA ENGINE v1.0", color=(100, 255, 150))
        dpg.add_spacer(width=20)
        dpg.add_text("|  STATUS: ")
        dpg.add_text(tag="status_text", default_value="DISCONNECTED", color=(255, 0, 0))
        dpg.add_spacer(width=20)
        dpg.add_text("|  MODEL: KNN-CLASSIFIER (K=3)")

    dpg.add_separator()
    dpg.add_spacer(height=5)

    with dpg.group(horizontal=True):
        # LEFT PANEL
        with dpg.child_window(width=280, border=True):
            dpg.add_text("SYSTEM CONTROL", bullet=True, color=(100, 255, 150))
            # Pasang callback ke tombol
            dpg.add_button(label="START ENGINE", width=-1, height=30, callback=engine_control, user_data="START")
            dpg.add_button(label="TERMINATE", width=-1, callback=engine_control, user_data="TERMINATE")

            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)

            dpg.add_text("ALGORITHM CONFIG", bullet=True, color=(100, 255, 150))
            dpg.add_slider_int(label="K-Neighbors", default_value=3, min_value=1, max_value=15)
            dpg.add_slider_float(label="Threshold", default_value=0.75, min_value=0.0, max_value=1.0)
            dpg.add_checkbox(label="Show Landmarks", default_value=True, tag="show_lm_cb")

            dpg.add_spacer(height=10)
            dpg.add_text("CALIBRATION", bullet=True, color=(100, 255, 150))
            dpg.add_combo(["Static Mode", "Dynamic Flow"], default_value="Static Mode", label="Input Type")
            dpg.add_button(label="Reset Coordinates", width=-1)

        # CENTER PANEL
        with dpg.group():
            with dpg.child_window(width=660, height=520, border=True):
                dpg.add_text("CAMERA OUTPUT", color=(150, 150, 150))
                dpg.add_text("Tekan tombol 'C' pada keyboard untuk Capture matrix data.", color=(200, 200, 100))
                dpg.add_spacer(height=5)
                dpg.add_image("camera_texture")

            with dpg.child_window(width=660, height=-1, border=True, tag="log_window"):
                dpg.add_text("SYSTEM LOGS", color=(100, 255, 150))
                with dpg.group(tag="log_group"):
                    dpg.add_text("[INFO] Gesture Engine Initialized...", color=(150, 150, 150))

        # RIGHT PANEL
        with dpg.child_window(width=-1, border=True):
            dpg.add_text("PREDICTION ANALYSIS", bullet=True, color=(100, 255, 150))
            dpg.add_text("Confidence Score:")
            dpg.add_progress_bar(label="Class A", default_value=0.85, overlay="Class A: 85%", width=-1)

            dpg.add_spacer(height=20)
            dpg.add_text("COORDINATE MATRIX", bullet=True, color=(100, 255, 150))
            with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
                dpg.add_table_column(label="Point")
                dpg.add_table_column(label="X")
                dpg.add_table_column(label="Y")
                for i in range(5):
                    with dpg.table_row():
                        dpg.add_text(f"LM {i}")
                        dpg.add_text("0.00", tag=f"t_x_{i}")
                        dpg.add_text("0.00", tag=f"t_y_{i}")

dpg.bind_theme(global_theme)
dpg.create_viewport(title="Gestura Engine - Technical Workspace", width=1280, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("PrimaryWindow", True)

# ================== CUSTOM RENDER LOOP ==================
capture_cooldown = 0

while dpg.is_dearpygui_running():
    # Hanya baca frame jika engine sedang running
    if engine_running and cap is not None:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            points, hand_landmarks = get_hand_points_mediapipe(frame)

            show_landmarks = dpg.get_value("show_lm_cb")

            if points is not None:
                if show_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    for i, (x, y) in enumerate(points):
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

                # Update tabel (5 titik pertama)
                for i in range(5):
                    lm = hand_landmarks.landmark[i]
                    dpg.set_value(f"t_x_{i}", f"{lm.x:.3f}")
                    dpg.set_value(f"t_y_{i}", f"{lm.y:.3f}")

                # Tombol 'C' (67 adalah kode key untuk 'C' di DPG)
                if dpg.is_key_down(67) and capture_cooldown == 0:
                    A = points.flatten().reshape(1, -1)
                    define_conture(A)
                    capture_cooldown = 15 # Cegah multiple capture

            if capture_cooldown > 0:
                capture_cooldown -= 1

            # Update Tekstur DPG
            rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            # Resize jika resolusi kamera tidak sama dengan canvas default
            if rgba_frame.shape[1] != cam_width or rgba_frame.shape[0] != cam_height:
                 rgba_frame = cv2.resize(rgba_frame, (cam_width, cam_height))
                 
            texture_data = rgba_frame.astype(np.float32) / 255.0
            dpg.set_value("camera_texture", texture_data.flatten()) # Pastikan data di-flatten

    # Render frame UI (Kamera hidup atau mati, UI harus tetap jalan)
    dpg.render_dearpygui_frame()

# ================== CLEANUP ==================
if cap is not None:
    cap.release()
hands.close()
dpg.destroy_context()