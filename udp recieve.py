import sys
import numpy as np
import math
import csv
import socket
import threading
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QComboBox, QTextEdit,
                             QHBoxLayout, QGroupBox, QRadioButton, QTableWidget, QTableWidgetItem, QScrollArea, QCheckBox)
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib.pyplot as plt
import mplcursors
from scipy.optimize import linear_sum_assignment

# Custom stream class to redirect stdout
class OutputStream:
    def __init__(self, text_edit):
        self.text_edit = text_edit

    def write(self, text):
        self.text_edit.append(text)

    def flush(self): 
        pass  # No need to implement flush for QTextEdit

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.Sp = np.zeros((6, 1))  # Predicted state vector
        self.Pp = np.eye(6)  # Predicted state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.prev_Time = 0
        self.Q = np.eye(6)
        self.Phi = np.eye(6)
        self.Z = np.zeros((3, 1))
        self.Z1 = np.zeros((3, 1))  # Measurement vector
        self.Z2 = np.zeros((3, 1))
        self.first_rep_flag = False
        self.second_rep_flag = False
        self.gate_threshold = 900.21  # 95% confidence interval for Chi-squared distribution with 3 degrees of freedom

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        print(f"Initializing filter state with x: {x}, y: {y}, z: {z}, vx: {vx}, vy: {vy}, vz: {vz}, time: {time}")
        if not self.first_rep_flag:
            self.Z1 = np.array([[x], [y], [z]])
            self.Sf[0] = x
            self.Sf[1] = y
            self.Sf[2] = z
            self.Meas_Time = time
            self.prev_Time = self.Meas_Time
            self.first_rep_flag = True
        elif self.first_rep_flag and not self.second_rep_flag:
            self.Z2 = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time
            dt = self.Meas_Time - self.prev_Time
            self.Sf[3] = (self.Z2[0] - self.Z1[0]) / dt
            self.Sf[4] = (self.Z2[1] - self.Z1[1]) / dt
            self.Sf[5] = (self.Z2[2] - self.Z1[2]) / dt
            self.second_rep_flag = True
        else:
            self.Z = np.array([[x], [y], [z]])
            self.prev_Time = self.Meas_Time
            self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.prev_Time
        T_2 = (dt * dt) / 2.0
        T_3 = (dt * dt * dt) / 3.0
        self.Phi[0, 3] = dt
        self.Phi[1, 4] = dt
        self.Phi[2, 5] = dt
        self.Q[0, 0] = T_3
        self.Q[1, 1] = T_3
        self.Q[2, 2] = T_3
        self.Q[0, 3] = T_2
        self.Q[1, 4] = T_2
        self.Q[2, 5] = T_2
        self.Q[3, 0] = T_2
        self.Q[4, 1] = T_2
        self.Q[5, 2] = T_2
        self.Q[3, 3] = dt
        self.Q[4, 4] = dt
        self.Q[5, 5] = dt
        self.Q = self.Q * self.plant_noise
        self.Sp = np.dot(self.Phi, self.Sf)
        self.Pp = np.dot(np.dot(self.Phi, self.Pf), self.Phi.T) + self.Q
        self.Meas_Time = current_time

    def update_step(self, Z):
        Inn = Z - np.dot(self.H, self.Sp)
        S = np.dot(self.H, np.dot(self.Pp, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pp, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sp + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pp)

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
    az = math.atan2(y, x)

    if x > 0.0:
        az = np.pi / 2 - az
    else:
        az = 3 * np.pi / 2 - az

    az = az * 180 / np.pi

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    return r, az, el

def form_measurement_groups(measurements, max_time_diff=0.050):
    measurement_groups = []
    current_group = []
    base_time = measurements[0][3]

    for measurement in measurements:
        if measurement[3] - base_time <= max_time_diff:
            current_group.append(measurement)
        else:
            measurement_groups.append(current_group)
            current_group = [measurement]
            base_time = measurement[3]

    if current_group:
        measurement_groups.append(current_group)

    return measurement_groups

def form_clusters_via_association(tracks, reports, kalman_filter):
    association_list = []
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])  # 3x3 covariance matrix for position only
    chi2_threshold = kalman_filter.gate_threshold

    for i, track in enumerate(tracks):
        for j, report in enumerate(reports):
            distance = mahalanobis_distance(track, report, cov_inv)
            if distance < chi2_threshold:
                association_list.append((i, j))

    clusters = []
    while association_list:
        cluster_tracks = set()
        cluster_reports = set()
        stack = [association_list.pop(0)]

        while stack:
            track_idx, report_idx = stack.pop()
            cluster_tracks.add(track_idx)
            cluster_reports.add(report_idx)
            new_assoc = [(t, r) for t, r in association_list if t == track_idx or r == report_idx]
            for assoc in new_assoc:
                if assoc not in stack:
                    stack.append(assoc)
            association_list = [assoc for assoc in association_list if assoc not in new_assoc]

        clusters.append((list(cluster_tracks), [reports[r] for r in cluster_reports]))

    return clusters

def mahalanobis_distance(track, report, cov_inv):
    residual = np.array(report) - np.array(track)
    distance = np.dot(np.dot(residual.T, cov_inv), residual)
    return distance

def select_best_report(cluster_tracks, cluster_reports, kalman_filter):
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])

    best_report = None
    best_track_idx = None
    max_weight = -np.inf

    for i, track in enumerate(cluster_tracks):
        for j, report in enumerate(cluster_reports):
            residual = np.array(report) - np.array(track)
            weight = np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
            if weight > max_weight:
                max_weight = weight
                best_report = report
                best_track_idx = i

    return best_track_idx, best_report

def select_initiation_mode(mode):
    if mode == '3-state':
        return 3
    elif mode == '5-state':
        return 5
    elif mode == '7-state':
        return 7
    else:
        raise ValueError("Invalid mode selected.")

def doppler_correlation(doppler_1, doppler_2, doppler_threshold):
    return abs(doppler_1 - doppler_2) < doppler_threshold

def correlation_check(track, measurement, doppler_threshold, range_threshold):
    last_measurement = track['measurements'][-1][0]
    last_cartesian = sph2cart(last_measurement[0], last_measurement[1], last_measurement[2])
    measurement_cartesian = sph2cart(measurement[0], measurement[1], measurement[2])
    distance = np.linalg.norm(np.array(measurement_cartesian) - np.array(last_cartesian))

    doppler_correlated = doppler_correlation(measurement[4], last_measurement[4], doppler_threshold)
    range_satisfied = distance < range_threshold

    return doppler_correlated and range_satisfied

def initialize_filter_state(kalman_filter, x, y, z, vx, vy, vz, time):
    kalman_filter.initialize_filter_state(x, y, z, vx, vy, vz, time)

def perform_jpda(tracks, reports, kalman_filter):
    clusters = form_clusters_via_association(tracks, reports, kalman_filter)
    best_reports = []
    hypotheses = []
    probabilities = []

    for cluster_tracks, cluster_reports in clusters:
        # Generate hypotheses for each cluster
        cluster_hypotheses = []
        cluster_probabilities = []
        for track in cluster_tracks:
            for report in cluster_reports:
                # Calculate the probability of the hypothesis
                cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])
                residual = np.array(report) - np.array(track)
                probability = np.exp(-0.5 * np.dot(np.dot(residual.T, cov_inv), residual))
                cluster_hypotheses.append((track, report))
                cluster_probabilities.append(probability)

        # Normalize probabilities
        total_probability = sum(cluster_probabilities)
        cluster_probabilities = [p / total_probability for p in cluster_probabilities]

        # Select the best hypothesis based on the highest probability
        best_hypothesis_index = np.argmax(cluster_probabilities)
        best_track, best_report = cluster_hypotheses[best_hypothesis_index]

        best_reports.append((best_track, best_report))
        hypotheses.append(cluster_hypotheses)
        probabilities.append(cluster_probabilities)

    # Log clusters, hypotheses, and probabilities
    print("JPDA Clusters:", clusters)
    print("JPDA Hypotheses:", hypotheses)
    print("JPDA Probabilities:", probabilities)
    print("JPDA Best Reports:", best_reports)

    return clusters, best_reports, hypotheses, probabilities

def perform_munkres(tracks, reports, kalman_filter):
    cost_matrix = []
    cov_inv = np.linalg.inv(kalman_filter.Pp[:3, :3])

    for track in tracks:
        track_costs = []
        for report in reports:
            distance = mahalanobis_distance(track, report, cov_inv)
            track_costs.append(distance)
        cost_matrix.append(track_costs)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    best_reports = [(row, reports[col]) for row, col in zip(row_ind, col_ind)]

    # Log cost matrix and assignments
    print("Munkres Cost Matrix:", cost_matrix)
    print("Munkres Assignments:", list(zip(row_ind, col_ind)))
    print("Munkres Best Reports:", best_reports)

    return best_reports

def check_track_timeout(tracks, current_time, poss_timeout=20.0, firm_tent_timeout=50.0):
    tracks_to_remove = []
    for track_id, track in enumerate(tracks):
        last_measurement_time = track['measurements'][-1][0][3]
        time_since_last_measurement = current_time - last_measurement_time

        if track['current_state'] == 'Poss1' and time_since_last_measurement > poss_timeout:
            tracks_to_remove.append(track_id)
        elif track['current_state'] in ['Tentative1', 'Firm'] and time_since_last_measurement > firm_tent_timeout:
            tracks_to_remove.append(track_id)

    return tracks_to_remove

def log_to_csv(log_file_path, data):
    with open(log_file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data.keys())
        writer.writerow(data)

class KalmanFilterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.tracks = []
        self.selected_track_ids = set()
        self.initUI()
        self.control_panel_collapsed = False

        # Start UDP receiver thread
        self.udp_thread = threading.Thread(target=self.receive_udp_data)
        self.udp_thread.daemon = True
        self.udp_thread.start()

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
        main_layout = QVBoxLayout()

        # Output Display
        self.output_display = QTextEdit()
        self.output_display.setFont(QFont('Courier', 10))
        self.output_display.setStyleSheet("background-color: #333333; color: #ffffff;")
        self.output_display.setReadOnly(True)
        main_layout.addWidget(self.output_display)

        # Plot Setup
        self.canvas = FigureCanvas(plt.Figure())
        main_layout.addWidget(self.canvas)

        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        main_layout.addWidget(self.toolbar)

        # Redirect stdout to the output display
        sys.stdout = OutputStream(self.output_display)

        # Set main layout
        self.setLayout(main_layout)

    def receive_udp_data(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", 5005))
        while True:
            data, addr = sock.recvfrom(1024)
            measurement_group = eval(data.decode('utf-8'))
            self.process_measurement_group(measurement_group)

    def process_measurement_group(self, measurement_group):
        if len(measurement_group) == 1:
            self.process_single_measurement(measurement_group[0])
        else:
            self.process_multiple_measurements(measurement_group)

    def process_single_measurement(self, measurement):
        # Initialize necessary variables
        doppler_threshold = 100
        range_threshold = 100
        firm_threshold = select_initiation_mode(track_mode)
        association_method = association_type  # 'JPDA' or 'Munkres'

        # Initialize variables outside the loop
        miss_counts = {}
        hit_counts = {}
        firm_ids = set()
        state_map = {}
        state_transition_times = {}
        progression_states = {
            3: ['Poss1', 'Tentative1', 'Firm'],
            5: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm'],
            7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Tentative3', 'Firm']
        }[firm_threshold]

        current_time = measurement[3]  # Assuming the time is at index 3 of each measurement
        assigned = False

        for track_id, track in enumerate(self.tracks):
            if correlation_check(track, measurement, doppler_threshold, range_threshold):
                current_state = state_map.get(track_id, None)
                if current_state == 'Poss1':
                    initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])
                elif current_state == 'Tentative1':
                    last_measurement = track['measurements'][-1][0]
                    dt = measurement[3] - last_measurement[3]
                    vx = (sph2cart(*measurement[:3])[0] - sph2cart(*last_measurement[:3])[0]) / dt
                    vy = (sph2cart(*measurement[:3])[1] - sph2cart(*last_measurement[:3])[1]) / dt
                    vz = (sph2cart(*measurement[:3])[2] - sph2cart(*last_measurement[:3])[2]) / dt
                    initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), vx, vy, vz, measurement[3])
                elif current_state == 'Firm':
                    kalman_filter.predict_step(measurement[3])
                    kalman_filter.update_step(np.array((measurement[:3])).reshape(3, 1))

                track['measurements'].append((measurement, current_state))
                track['Sf'].append(kalman_filter.Sf.copy())
                track['Sp'].append(kalman_filter.Sp.copy())
                track['Pp'].append(kalman_filter.Pp.copy())
                track['Pf'].append(kalman_filter.Pf.copy())
                hit_counts[track_id] = hit_counts.get(track_id, 0) + 1
                assigned = True

                # Log data to CSV
                log_data = {
                    'Time': measurement[3],
                    'Measurement X': measurement[5],
                    'Measurement Y': measurement[6],
                    'Measurement Z': measurement[7],
                    'Current State': current_state,
                    'Correlation Output': 'Yes',
                    'Associated Track ID': track_id,
                    'Associated Position X': track['Sf'][-1][0, 0],
                    'Associated Position Y': track['Sf'][-1][1, 0],
                    'Associated Position Z': track['Sf'][-1][2, 0],
                    'Association Type': 'Single',
                    'Clusters Formed': '',
                    'Hypotheses Generated': '',
                    'Probability of Hypothesis': '',
                    'Best Report Selected': ''
                }
                log_to_csv(log_file_path, log_data)
                break

        if not assigned:
            new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
            if new_track_id is None:
                new_track_id = len(track_id_list)
                track_id_list.append({'id': new_track_id, 'state': 'occupied'})
            else:
                track_id_list[new_track_id]['state'] = 'occupied'

            self.tracks.append({
                'track_id': new_track_id,
                'measurements': [(measurement, 'Poss1')],
                'current_state': 'Poss1',
                'Sf': [kalman_filter.Sf.copy()],
                'Sp': [kalman_filter.Sp.copy()],
                'Pp': [kalman_filter.Pp.copy()],
                'Pf': [kalman_filter.Pf.copy()]
            })
            state_map[new_track_id] = 'Poss1'
            state_transition_times[new_track_id] = {'Poss1': current_time}
            hit_counts[new_track_id] = 1
            initialize_filter_state(kalman_filter, *sph2cart(*measurement[:3]), 0, 0, 0, measurement[3])

            # Log data to CSV
            log_data = {
                'Time': measurement[3],
                'Measurement X': measurement[5],
                'Measurement Y': measurement[6],
                'Measurement Z': measurement[7],
                'Current State': 'Poss1',
                'Correlation Output': 'No',
                'Associated Track ID': new_track_id,
                'Associated Position X': '',
                'Associated Position Y': '',
                'Associated Position Z': '',
                'Association Type': 'New',
                'Clusters Formed': '',
                'Hypotheses Generated': '',
                'Probability of Hypothesis': '',
                'Best Report Selected': ''
            }
            log_to_csv(log_file_path, log_data)

    def process_multiple_measurements(self, measurements):
        # Initialize necessary variables
        doppler_threshold = 100
        range_threshold = 100
        firm_threshold = select_initiation_mode(track_mode)
        association_method = association_type  # 'JPDA' or 'Munkres'

        # Initialize variables outside the loop
        miss_counts = {}
        hit_counts = {}
        firm_ids = set()
        state_map = {}
        state_transition_times = {}
        progression_states = {
            3: ['Poss1', 'Tentative1', 'Firm'],
            5: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Firm'],
            7: ['Poss1', 'Poss2', 'Tentative1', 'Tentative2', 'Tentative3', 'Firm']
        }[firm_threshold]

        current_time = measurements[0][3]  # Assuming the time is at index 3 of each measurement
        reports = [sph2cart(*m[:3]) for m in measurements]

        if association_method == 'JPDA':
            clusters, best_reports, hypotheses, probabilities = perform_jpda(
                [track['measurements'][-1][0][:3] for track in self.tracks], reports, kalman_filter
            )
        elif association_method == 'Munkres':
            best_reports = perform_munkres([track['measurements'][-1][0][:3] for track in self.tracks], reports, kalman_filter)

        for track_id, best_report in best_reports:
            current_state = state_map.get(track_id, None)
            if current_state == 'Poss1':
                initialize_filter_state(kalman_filter, *best_report, 0, 0, 0, current_time)
            elif current_state == 'Tentative1':
                last_measurement = self.tracks[track_id]['measurements'][-1][0]
                dt = current_time - last_measurement[3]
                vx = (best_report[0] - sph2cart(*last_measurement[:3])[0]) / dt
                vy = (best_report[1] - sph2cart(*last_measurement[:3])[1]) / dt
                vz = (best_report[2] - sph2cart(*last_measurement[:3])[2]) / dt
                initialize_filter_state(kalman_filter, *best_report, vx, vy, vz, current_time)
            elif current_state == 'Firm':
                kalman_filter.predict_step(current_time)
                kalman_filter.update_step(np.array(best_report).reshape(3, 1))

            self.tracks[track_id]['measurements'].append((cart2sph(*best_report) + (current_time, measurements[0][4]), current_state))
            self.tracks[track_id]['Sf'].append(kalman_filter.Sf.copy())
            self.tracks[track_id]['Sp'].append(kalman_filter.Sp.copy())
            self.tracks[track_id]['Pp'].append(kalman_filter.Pp.copy())
            self.tracks[track_id]['Pf'].append(kalman_filter.Pf.copy())
            hit_counts[track_id] = hit_counts.get(track_id, 0) + 1

            # Log data to CSV
            log_data = {
                'Time': current_time,
                'Measurement X': best_report[0],
                'Measurement Y': best_report[1],
                'Measurement Z': best_report[2],
                'Current State': current_state,
                'Correlation Output': 'Yes',
                'Associated Track ID': track_id,
                'Associated Position X': self.tracks[track_id]['Sf'][-1][0, 0],
                'Associated Position Y': self.tracks[track_id]['Sf'][-1][1, 0],
                'Associated Position Z': self.tracks[track_id]['Sf'][-1][2, 0],
                'Association Type': association_method,
                'Hypotheses Generated': '',
                'Probability of Hypothesis': '',
                'Best Report Selected': best_report
            }
            log_to_csv(log_file_path, log_data)

        # Handle unassigned measurements
        assigned_reports = set(best_report for _, best_report in best_reports)
        for report in reports:
            if tuple(report) not in assigned_reports:
                new_track_id = next((i for i, t in enumerate(track_id_list) if t['state'] == 'free'), None)
                if new_track_id is None:
                    new_track_id = len(track_id_list)
                    track_id_list.append({'id': new_track_id, 'state': 'occupied'})
                else:
                    track_id_list[new_track_id]['state'] = 'occupied'

                self.tracks.append({
                    'track_id': new_track_id,
                    'measurements': [(cart2sph(*report) + (current_time, measurements[0][4]), 'Poss1')],
                    'current_state': 'Poss1',
                    'Sf': [kalman_filter.Sf.copy()],
                    'Sp': [kalman_filter.Sp.copy()],
                    'Pp': [kalman_filter.Pp.copy()],
                    'Pf': [kalman_filter.Pf.copy()]
                })
                state_map[new_track_id] = 'Poss1'
                state_transition_times[new_track_id] = {'Poss1': current_time}
                hit_counts[new_track_id] = 1
                initialize_filter_state(kalman_filter, *report, 0, 0, 0, current_time)

                # Log data to CSV
                log_data = {
                    'Time': current_time,
                    'Measurement X': report[0],
                    'Measurement Y': report[1],
                    'Measurement Z': report[2],
                    'Current State': 'Poss1',
                    'Correlation Output': 'No',
                    'Associated Track ID': new_track_id,
                    'Associated Position X': '',
                    'Associated Position Y': '',
                    'Associated Position Z': '',
                    'Association Type': 'New',
                    'Hypotheses Generated': '',
                    'Probability of Hypothesis': '',
                    'Best Report Selected': ''
                }
                log_to_csv(log_file_path, log_data)

        # Update states based on hit counts
        for track_id, track in enumerate(self.tracks):
            current_state = state_map.get(track_id, None)
            if current_state is not None:
                current_state_index = progression_states.index(current_state)
                if hit_counts[track_id] >= firm_threshold and current_state != 'Firm':
                    state_map[track_id] = 'Firm'
                    firm_ids.add(track_id)
                    state_transition_times.setdefault(track_id, {})['Firm'] = current_time
                elif current_state_index < len(progression_states) - 1:
                    next_state = progression_states[current_state_index + 1]
                    if hit_counts[track_id] >= current_state_index + 1 and state_map[track_id] != next_state:
                        state_map[track_id] = next_state
                        state_transition_times.setdefault(track_id, {})[next_state] = current_time
                track['current_state'] = state_map[track_id]

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = KalmanFilterGUI()
    ex.show()
    sys.exit(app.exec_())
