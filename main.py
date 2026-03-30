import dearpygui.dearpygui as dpg

dpg.create_context()

# --- 1. TECHNICAL THEME (DARK MODE & ACCENTS) ---
with dpg.theme() as global_theme:
    with dpg.theme_component(dpg.mvAll):
        # Window & Panel Colors
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (15, 15, 15), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (24, 24, 28), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Border, (40, 40, 45), category=dpg.mvThemeCat_Core)
        
        # Elements (Buttons, Sliders, etc)
        dpg.add_theme_color(dpg.mvThemeCol_Button, (45, 45, 55), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (100, 255, 150, 150), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_Header, (40, 40, 50), category=dpg.mvThemeCat_Core)
        dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (100, 255, 150), category=dpg.mvThemeCat_Core)
        
        # Rounding for professional look
        dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 3)
        dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
        dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 10)

# --- 2. LAYOUT UTAMA (WORKSPACE MODE) ---
with dpg.window(tag="PrimaryWindow"):
    
    # TOP HEADER / STATUS BAR
    with dpg.group(horizontal=True):
        dpg.add_text(" GESTURA ENGINE v1.0.4", color=(100, 255, 150))
        dpg.add_spacer(width=20)
        dpg.add_text("|  STATUS: ")
        dpg.add_text("CONNECTED", color=(0, 255, 0))
        dpg.add_spacer(width=20)
        dpg.add_text("|  MODEL: KNN-CLASSIFIER (K=3)")

    dpg.add_separator()
    dpg.add_spacer(height=5)

    with dpg.group(horizontal=True):
        
        # --- LEFT PANEL: CONTROL & PARAMETERS (25% Width) ---
        with dpg.child_window(width=280, border=True):
            dpg.add_text("SYSTEM CONTROL", bullet=True, color=(100, 255, 150))
            dpg.add_button(label="START ENGINE", width=-1, height=30)
            dpg.add_button(label="TERMINATE", width=-1)
            
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=10)
            
            dpg.add_text("ALGORITHM CONFIG", bullet=True, color=(100, 255, 150))
            dpg.add_slider_int(label="K-Neighbors", default_value=3, min_value=1, max_value=15)
            dpg.add_slider_float(label="Threshold", default_value=0.75, min_value=0.0, max_value=1.0)
            dpg.add_checkbox(label="Show Landmarks")
            dpg.add_checkbox(label="Show Bounding Box", default_value=True)
            
            dpg.add_spacer(height=10)
            dpg.add_text("CALIBRATION", bullet=True, color=(100, 255, 150))
            dpg.add_combo(["Static Mode", "Dynamic Flow"], default_value="Static Mode", label="Input Type")
            dpg.add_button(label="Reset Coordinates", width=-1)

        # --- CENTER PANEL: LIVE FEED (50% Width) ---
        with dpg.group():
            with dpg.child_window(width=600, height=450, border=True):
                # Ini tempat video OpenCV nantinya
                dpg.add_text("LIVE STREAMING OUTPUT", color=(150, 150, 150))
                dpg.add_spacer(height=180)
                with dpg.group(horizontal=True):
                    dpg.add_spacer(width=200)
                    dpg.add_loading_indicator(style=1)
            
            # BOTTOM PANEL: LIVE CONSOLE LOG
            with dpg.child_window(width=600, height=-1, border=True):
                dpg.add_text("SYSTEM LOGS", color=(100, 255, 150))
                dpg.add_text("[INFO] Gesture Engine Initialized...", color=(150, 150, 150))
                dpg.add_text("[INFO] KNN Model Loaded: 14 classes found", color=(150, 150, 150))
                dpg.add_text("[DATA] Hand detected at (x:342, y:112)", color=(200, 200, 200))

        # --- RIGHT PANEL: ANALYTICS & PROBABILITY (25% Width) ---
        with dpg.child_window(width=-1, border=True):
            dpg.add_text("PREDICTION ANALYSIS", bullet=True, color=(100, 255, 150))
            
            # Simulasi Grafik Probabilitas KNN
            dpg.add_text("Confidence Score:")
            dpg.add_progress_bar(label="Class A", default_value=0.85, overlay="Class A: 85%", width=-1)
            dpg.add_progress_bar(label="Class B", default_value=0.12, overlay="Class B: 12%", width=-1)
            dpg.add_progress_bar(label="Others", default_value=0.03, overlay="Others: 3%", width=-1)
            
            dpg.add_spacer(height=20)
            dpg.add_text("COORDINATE MATRIX", bullet=True, color=(100, 255, 150))
            # Tabel data teknis
            with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
                dpg.add_table_column(label="Point")
                dpg.add_table_column(label="X")
                dpg.add_table_column(label="Y")
                for i in range(5):
                    with dpg.table_row():
                        dpg.add_text(f"Landmark {i}")
                        dpg.add_text("0.452")
                        dpg.add_text("0.891")

dpg.bind_theme(global_theme)

# --- 3. VIEWPORT CONFIG ---
dpg.create_viewport(title='Gestura Engine - Technical Workspace', width=1200, height=800)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("PrimaryWindow", True)
dpg.start_dearpygui()
dpg.destroy_context()