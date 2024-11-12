class SystemConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("System Configuration")
        self.setGeometry(300, 300, 350, 350)

        grid = QGridLayout()

        # Target Speed
        self.target_speed_group = QGroupBox("Target Speed (m/s)")
        speed_layout = QHBoxLayout()
        self.min_speed_edit = QLineEdit()
        self.min_speed_edit.setPlaceholderText("Min")
        speed_layout.addWidget(self.min_speed_edit)
        self.max_speed_edit = QLineEdit()
        self.max_speed_edit.setPlaceholderText("Max")
        speed_layout.addWidget(self.max_speed_edit)
        self.target_speed_group.setLayout(speed_layout)
        grid.addWidget(self.target_speed_group, 0, 0, 1, 2)

        # Target Altitude
        self.target_altitude_group = QGroupBox("Target Altitude (m)")
        altitude_layout = QHBoxLayout()
        self.min_altitude_edit = QLineEdit()
        self.min_altitude_edit.setPlaceholderText("Min")
        altitude_layout.addWidget(self.min_altitude_edit)
        self.max_altitude_edit = QLineEdit()
        self.max_altitude_edit.setPlaceholderText("Max")
        altitude_layout.addWidget(self.max_altitude_edit)
        self.target_altitude_group.setLayout(altitude_layout)
        grid.addWidget(self.target_altitude_group, 1, 0, 1, 2)

        # Correlation Gates
        self.correlation_gates_group = QGroupBox("Correlation Gates")
        gates_layout = QGridLayout()
        self.range_gate_group = QGroupBox("Range Gate (m)")
        range_layout = QHBoxLayout()
        self.min_range_edit = QLineEdit()
        self.min_range_edit.setPlaceholderText("Min")
        range_layout.addWidget(self.min_range_edit)
        self.max_range_edit = QLineEdit()
        self.max_range_edit.setPlaceholderText("Max")
        range_layout.addWidget(self.max_range_edit)
        self.range_gate_group.setLayout(range_layout)
        gates_layout.addWidget(self.range_gate_group, 0, 0)

        self.azimuth_gate_group = QGroupBox("Azimuth Gate (°)")
        azimuth_layout = QHBoxLayout()
        self.min_azimuth_edit = QLineEdit()
        self.min_azimuth_edit.setPlaceholderText("Min")
        azimuth_layout.addWidget(self.min_azimuth_edit)
        self.max_azimuth_edit = QLineEdit()
        self.max_azimuth_edit.setPlaceholderText("Max")
        azimuth_layout.addWidget(self.max_azimuth_edit)
        self.azimuth_gate_group.setLayout(azimuth_layout)
        gates_layout.addWidget(self.azimuth_gate_group, 1, 0)

        self.elevation_gate_group = QGroupBox("Elevation Gate (°)")
        elevation_layout = QHBoxLayout()
        self.min_elevation_edit = QLineEdit()
        self.min_elevation_edit.setPlaceholderText("Min")
        elevation_layout.addWidget(self.min_elevation_edit)
        self.max_elevation_edit = QLineEdit()
        self.max_elevation_edit.setPlaceholderText("Max")
        elevation_layout.addWidget(self.max_elevation_edit)
        self.elevation_gate_group.setLayout(elevation_layout)
        gates_layout.addWidget(self.elevation_gate_group, 2, 0)

        self.correlation_gates_group.setLayout(gates_layout)
        grid.addWidget(self.correlation_gates_group, 2, 0, 3, 2)

        # Plant Noise
        self.plant_noise_label = QLabel("Plant Noise Covariance:")
        self.plant_noise_edit = QLineEdit()
        grid.addWidget(self.plant_noise_label, 5, 0)
        grid.addWidget(self.plant_noise_edit, 5, 1)

        # OK and Cancel buttons
        button_box = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_box.addWidget(ok_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_box.addWidget(cancel_button)
        grid.addLayout(button_box, 6, 0, 1, 2)

        self.setLayout(grid)

    def get_config_data(self):
        return {
            "target_speed": (float(self.min_speed_edit.text()), float(self.max_speed_edit.text())),
            "target_altitude": (float(self.min_altitude_edit.text()), float(self.max_altitude_edit.text())),
            "range_gate": (float(self.min_range_edit.text()), float(self.max_range_edit.text())),
            "azimuth_gate": (float(self.min_azimuth_edit.text()), float(self.max_azimuth_edit.text())),
            "elevation_gate": (float(self.min_elevation_edit.text()), float(self.max_elevation_edit.text())),
            "plant_noise": float(self.plant_noise_edit.text())
        }


class Signal(QObject):
    # Signal for collapsing the control panel
    collapseSignal = pyqtSignal(bool)

class KalmanFilterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.tracks = []
        self.selected_track_ids = set()
        self.initUI()
        self.control_panel_collapsed = False  # Start with the panel expanded

    def initUI(self):
        self.setWindowTitle('Kalman Filter GUI')
        self.setGeometry(100, 100, 1200, 600)
        self.setStyleSheet("""
            QWidget {
                background-color: #222222;
                color: #ffffff;
                font-family: "Arial", sans-serif;
            }
            QPushButton {
                background-color: #4CAF50; 
                color: white;
                border: none;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 16px;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3e8e41;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QComboBox {
                background-color: #222222;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
            QLineEdit {
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
            QRadioButton {
                background-color: transparent;
                color: white;
            }
            QTextEdit {
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
            QGroupBox {
                background-color: #333333;
                border: 1px solid #555555;
                border-radius: 4px;
                padding: 5px;
            }
            QTableWidget {
                background-color: #333333;
                color: white;
                border: 1px solid #555555;
                font-size: 12px;
            }
        """)

        # Main layout
        main_layout = QHBoxLayout()

        # Left side: System Configuration and Controls (Collapsible)
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)

        # Collapse/Expand Button
        self.collapse_button = QToolButton()
        self.collapse_button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.collapse_button.setText("=")  # Set the button text to "="
        self.collapse_button.clicked.connect(self.toggle_control_panel)
        left_layout.addWidget(self.collapse_button)

        # Control Panel
        self.control_panel = QWidget()
        self.control_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        control_layout = QVBoxLayout()
        self.control_panel.setLayout(control_layout)
        left_layout.addWidget(self.control_panel)

        # File Upload Button
        self.file_upload_button = QPushButton("Upload File")
        self.file_upload_button.setIcon(QIcon("upload.png"))
        self.file_upload_button.clicked.connect(self.select_file)
        control_layout.addWidget(self.file_upload_button)

        # System Configuration button
        self.config_button = QPushButton("System Configuration")
        self.config_button.setIcon(QIcon("config.png"))
        self.config_button.clicked.connect(self.show_config_dialog)
        control_layout.addWidget(self.config_button)

        # Initiate Track drop down
        self.track_mode_label = QLabel("Initiate Track")
        self.track_mode_combo = QComboBox()
        self.track_mode_combo.addItems(["3-state", "5-state", "7-state"])
        control_layout.addWidget(self.track_mode_label)
        control_layout.addWidget(self.track_mode_combo)

        # Association Technique radio buttons
        self.association_group = QGroupBox("Association Technique")
        association_layout = QVBoxLayout()
        self.jpda_radio = QRadioButton("JPDA")
        self.jpda_radio.setChecked(True)
        association_layout.addWidget(self.jpda_radio)
        self.munkres_radio = QRadioButton("Munkres")
        association_layout.addWidget(self.munkres_radio)
        self.association_group.setLayout(association_layout)
        control_layout.addWidget(self.association_group)

        # Filter modes buttons
        self.filter_group = QGroupBox("Filter Modes")
        filter_layout = QHBoxLayout()
        self.cv_filter_button = QPushButton("CV Filter")
        filter_layout.addWidget(self.cv_filter_button)
        self.ca_filter_button = QPushButton("CA Filter")
        filter_layout.addWidget(self.ca_filter_button)
        self.ct_filter_button = QPushButton("CT Filter")
        filter_layout.addWidget(self.ct_filter_button)
        self.filter_group.setLayout(filter_layout)
        control_layout.addWidget(self.filter_group)

        # Plot Type dropdown
        self.plot_type_label = QLabel("Plot Type")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Range vs Time", "Azimuth vs Time", "Elevation vs Time", "PPI", "RHI", "All Modes"])
        control_layout.addWidget(self.plot_type_label)
        control_layout.addWidget(self.plot_type_combo)

        # Process button
        self.process_button = QPushButton("Process")
        self.process_button.setIcon(QIcon("process.png"))
        self.process_button.clicked.connect(self.process_data)
        control_layout.addWidget(self.process_button)

        # Right side: Output and Plot (with Tabs)
        right_layout = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # Tab Widget for Output, Plot, and Track Info
        self.tab_widget = QTabWidget()
        self.output_tab = QWidget()
        self.plot_tab = QWidget()
        self.track_info_tab = QWidget()  # New Track Info Tab
        self.tab_widget.addTab(self.output_tab, "Output")
        self.tab_widget.addTab(self.plot_tab, "Plot")
        self.tab_widget.addTab(self.track_info_tab, "Track Info")  # Add Track Info Tab
        self.tab_widget.setStyleSheet(" color: black;")
        right_layout.addWidget(self.tab_widget)

        # Output Display
        self.output_display = QTextEdit()
        self.output_display.setFont(QFont('Courier', 10))
        self.output_display.setStyleSheet("background-color: #333333; color: #ffffff;")
        self.output_display.setReadOnly(True)
        self.output_tab.setLayout(QVBoxLayout())
        self.output_tab.layout().addWidget(self.output_display)

        # Plot Setup
        self.canvas = FigureCanvas(plt.Figure())
        self.plot_tab.setLayout(QVBoxLayout())
        self.plot_tab.layout().addWidget(self.canvas)

        # Add navigation toolbar once
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.plot_tab.layout().addWidget(self.toolbar)

        # Add Clear Plot and Clear Output buttons
        self.clear_plot_button = QPushButton("Clear Plot")
        self.clear_plot_button.clicked.connect(self.clear_plot)
        self.plot_tab.layout().addWidget(self.clear_plot_button)

        self.clear_output_button = QPushButton("Clear Output")
        self.clear_output_button.clicked.connect(self.clear_output)
        self.output_tab.layout().addWidget(self.clear_output_button)

        # Track Info Setup
        self.track_info_layout = QVBoxLayout()
        self.track_info_tab.setLayout(self.track_info_layout)

        # Buttons to load CSV files
        self.load_detailed_log_button = QPushButton("Load Detailed Log")
        self.load_detailed_log_button.clicked.connect(lambda: self.load_csv('detailed_log.csv'))
        self.track_info_layout.addWidget(self.load_detailed_log_button)

        self.load_track_summary_button = QPushButton("Load Track Summary")
        self.load_track_summary_button.clicked.connect(lambda: self.load_csv('track_summary.csv'))
        self.track_info_layout.addWidget(self.load_track_summary_button)

        # Table to display CSV data
        self.csv_table = QTableWidget()
        self.csv_table.setStyleSheet("background-color: black; color: red;")  # Set text color to white
        self.track_info_layout.addWidget(self.csv_table)

        # Track ID Selection
        self.track_selection_group = QGroupBox("Select Track IDs to Plot")
        self.track_selection_layout = QVBoxLayout()
        self.track_selection_group.setLayout(self.track_selection_layout)
        self.plot_tab.layout().addWidget(self.track_selection_group)

        # Scroll area for track ID checkboxes
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.track_selection_widget = QWidget()
        self.track_selection_layout_inner = QVBoxLayout()
        self.track_selection_widget.setLayout(self.track_selection_layout_inner)
        self.scroll_area.setWidget(self.track_selection_widget)
        self.track_selection_layout.addWidget(self.scroll_area)

        main_layout.addWidget(right_widget)

        # Redirect stdout to the output display
        sys.stdout = OutputStream(self.output_display)

        # Set main layout
        self.setLayout(main_layout)

        # Initial settings
        self.config_data = {
            "target_speed": (0, 100),
            "target_altitude": (0, 10000),
            "range_gate": (0, 1000),
            "azimuth_gate": (0, 360),
            "elevation_gate": (0, 90),
            "plant_noise": 20  # Default value
        }

        # Add connections to filter buttons
        self.cv_filter_button.clicked.connect(lambda: self.select_filter("CV"))
        self.ca_filter_button.clicked.connect(lambda: self.select_filter("CA"))
        self.ct_filter_button.clicked.connect(lambda: self.select_filter("CT"))

        # Set initial filter mode
        self.filter_mode = "CV"  # Start with CV Filter
        self.update_filter_selection()

    def toggle_control_panel(self):
        self.control_panel_collapsed = not self.control_panel_collapsed
        self.control_panel.setVisible(not self.control_panel_collapsed)
        self.adjustSize()

    def select_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Input File", "", "CSV Files (*.csv);;All Files (*)", options=options
        )
        if file_name:
            self.input_file = file_name
            print(f"File selected: {self.input_file}")

    def process_data(self):
        input_file = getattr(self, "input_file", None)
        track_mode = self.track_mode_combo.currentText()
        association_type = "JPDA" if self.jpda_radio.isChecked() else "Munkres"
        filter_option = self.filter_mode

        if not input_file:
            print("Please select an input file.")
            return

        print(
            f"Processing with:\nInput File: {input_file}\nTrack Mode: {track_mode}\nFilter Option: {filter_option}\nAssociation Type: {association_type}"
        )

        self.tracks = main(
            input_file, track_mode, filter_option, association_type
        )  # Process data with selected parameters

        if self.tracks is None:
            print("No tracks were generated.")
        else:
            print(f"Number of tracks: {len(self.tracks)}")

            # Update the plot after processing
            self.update_plot()

            # Update track selection checkboxes
            self.update_track_selection()

    def update_plot(self):
        if not self.tracks:
            print("No tracks to plot.")
            return

        if len(self.tracks) == 0:
            print("Track list is empty.")
            return

        plot_type = self.plot_type_combo.currentText()

        self.canvas.figure.clear()  # Clear the canvas before plotting
        ax = self.canvas.figure.subplots()

        if plot_type == "All Modes":
            self.plot_all_modes(self.tracks, ax)
        elif plot_type == "PPI":
            self.plot_ppi(self.tracks, ax)
        elif plot_type == "RHI":
            self.plot_rhi(self.tracks, ax)
        else:
            plot_measurements(self.tracks, ax, plot_type, self.selected_track_ids)

        # Enable interactive data tips
        cursor = mplcursors.cursor(hover=True)

        @cursor.connect("add")
        def on_add(sel):
            index = sel.target.index
            track_id = self.tracks[index // len(self.tracks[0]['measurements'])]['track_id']
            measurement = self.tracks[index // len(self.tracks[0]['measurements'])]['measurements'][index % len(self.tracks[0]['measurements'])]
            time = measurement[0][3]
            sp = self.tracks[index // len(self.tracks[0]['measurements'])]['Sp']
            sf = self.tracks[index // len(self.tracks[0]['measurements'])]['Sf']
            plant_noise = 20

            sel.annotation.set(text=f"Track ID: {track_id}\nMeasurement: {measurement}\nTime: {time}\nSp: {sp}\nSf: {sf}\nPlant Noise: {plant_noise}")

        self.canvas.draw()

    def plot_all_modes(self, tracks, ax):
        # Create a 2x2 grid for subplots within the existing canvas
        self.canvas.figure.clear()
        axes = self.canvas.figure.subplots(2, 2)

        # Plot Range vs Time
        plot_measurements(tracks, axes[0, 0], "Range vs Time", self.selected_track_ids)
        axes[0, 0].set_title("Range vs Time")

        # Plot Azimuth vs Time
        plot_measurements(tracks, axes[0, 1], "Azimuth vs Time", self.selected_track_ids)
        axes[0, 1].set_title("Azimuth vs Time")

        # Plot PPI
        self.plot_ppi(tracks, axes[1, 0])
        axes[1, 0].set_title("PPI Plot")

        # Plot RHI
        self.plot_rhi(tracks, axes[1, 1])
        axes[1, 1].set_title("RHI Plot")

        # Adjust layout
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def plot_ppi(self, tracks, ax):
        ax.clear()
        for track in tracks:
            if track['track_id'] not in self.selected_track_ids:
                continue

            measurements = track["measurements"]
            x_coords = [sph2cart(*m[0][:3])[0] for m in measurements]
            y_coords = [sph2cart(*m[0][:3])[1] for m in measurements]

            # PPI plot (x vs y)
            ax.plot(x_coords, y_coords, label=f"Track {track['track_id']} PPI", marker="o")

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_title("PPI Plot (360°)")
        ax.legend()

    def plot_rhi(self, tracks, ax):
        ax.clear()
        for track in tracks:
            if track['track_id'] not in self.selected_track_ids:
                continue

            measurements = track["measurements"]
            x_coords = [sph2cart(*m[0][:3])[0] for m in measurements]
            z_coords = [sph2cart(*m[0][:3])[2] for m in measurements]

            # RHI plot (x vs z)
            ax.plot(
                x_coords, z_coords, label=f"Track {track['track_id']} RHI", linestyle="--"
            )

        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Z Coordinate")
        ax.set_title("RHI Plot")
        ax.legend()

    def show_config_dialog(self):
        dialog = SystemConfigDialog(self)
        if dialog.exec_():
            self.config_data = dialog.get_config_data()
            print(f"System Configuration Updated: {self.config_data}")

    def select_filter(self, filter_type):
        self.filter_mode = filter_type
        self.update_filter_selection()

    def update_filter_selection(self):
        self.cv_filter_button.setChecked(self.filter_mode == "CV")
        self.ca_filter_button.setChecked(self.filter_mode == "CA")
        self.ct_filter_button.setChecked(self.filter_mode == "CT")

    def clear_plot(self):
        self.canvas.figure.clear()
        self.canvas.draw()

    def clear_output(self):
        self.output_display.clear()

    def load_csv(self, file_path):
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                headers = next(reader)
                self.csv_table.setColumnCount(len(headers))
                self.csv_table.setHorizontalHeaderLabels(headers)

                # Clear existing rows
                self.csv_table.setRowCount(0)

                # Add rows from CSV
                for row_data in reader:
                    row = self.csv_table.rowCount()
                    self.csv_table.insertRow(row)
                    for column, data in enumerate(row_data):
                        self.csv_table.setItem(row, column, QTableWidgetItem(data))
        except Exception as e:
            print(f"Error loading CSV file: {e}")

    def update_track_selection(self):
        # Clear existing checkboxes
        for i in reversed(range(self.track_selection_layout_inner.count())):
            widget = self.track_selection_layout_inner.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Add "Select All" checkbox
        self.select_all_checkbox = QCheckBox("Select All Tracks")
        self.select_all_checkbox.setChecked(True)
        self.select_all_checkbox.stateChanged.connect(self.toggle_select_all_tracks)
        self.track_selection_layout_inner.addWidget(self.select_all_checkbox)

        # Add checkboxes for each track
        self.track_checkboxes = []
        for track in self.tracks:
            checkbox = QCheckBox(f"Track ID {track['track_id']}")
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.update_selected_tracks)
            self.track_selection_layout_inner.addWidget(checkbox)
            self.track_checkboxes.append(checkbox)

    def toggle_select_all_tracks(self, state):
        # Update all track checkboxes based on the "Select All" checkbox state
        for checkbox in self.track_checkboxes:
            checkbox.setChecked(state == Qt.Checked)

    def update_selected_tracks(self):
        self.selected_track_ids.clear()
        for checkbox in self.track_checkboxes:
            if checkbox.isChecked():
                track_id = int(checkbox.text().split()[-1])
                self.selected_track_ids.add(track_id)

        # Update the plot with selected tracks
        self.update_plot()

class NavigationToolbar(NavigationToolbar2QT):
    pass  # Use pass if there are no additional methods or attributes
integrate these functions also in my main code so it will work as per my current code and will perform all functionality 

so after updating and integrating give me a full code