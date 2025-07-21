import sys
import os
import numpy as np
import pyfftw
from PyQt5.QtGui import QDoubleValidator, QImage, QPixmap, QPainter, QPen, QConicalGradient
import numexpr as ne
import time
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSlider, QFileDialog, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRectF

class CircularProgressBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 100)
        self.setStyleSheet("background: transparent;")
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.angle_offset = 0
        self.animation_timer.start(30)  # ~33 FPS

    def start_animation(self):
        """Start the continuous rotation animation"""
        self.animation_timer.start(30)

    def stop_animation(self):
        """Stop the animation"""
        self.animation_timer.stop()

    def update_animation(self):
        self.angle_offset = (self.angle_offset + 8) % 360  # Increased rotation speed
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background circle
        pen_bg = QPen(Qt.blue, 6, Qt.SolidLine)
        painter.setPen(pen_bg)
        painter.drawEllipse(15, 15, 70, 70)

        # Draw animated rotating arc
        rect = QRectF(15, 15, 70, 70)
        start_angle = self.angle_offset * 16  # Start angle rotates
        span_angle = 120 * 16  # Fixed arc length (120 degrees)

        # Create a gradient for the rotating arc
        gradient = QConicalGradient(rect.center(), self.angle_offset)
        gradient.setColorAt(0.0, Qt.blue)
        gradient.setColorAt(0.5, Qt.cyan)
        gradient.setColorAt(1.0, Qt.blue)

        pen_progress = QPen()
        pen_progress.setWidth(6)
        pen_progress.setBrush(gradient)
        painter.setPen(pen_progress)
        painter.drawArc(rect, start_angle, span_angle)

class HologramLoadThread(QThread):
    loading_complete = pyqtSignal(object, str)  # hologram_data, error_message
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        
    def run(self):
        try:
            # Load the hologram
            hologram_loaded = plt.imread(self.file_path)
            
            if hologram_loaded.ndim == 3:
                hologram_loaded = np.mean(hologram_loaded, axis=2)
            
            hologram_data = hologram_loaded.astype(float)
            
            if hologram_data.size == 0:
                raise ValueError("Loaded hologram is empty.")
            
            # Small delay to show loading animation
            self.msleep(500)
            
            self.loading_complete.emit(hologram_data, "")
            
        except Exception as e:
            self.loading_complete.emit(None, str(e))

class HologramReconstructionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.hologram = None
        self.Nr0 = 0
        self.Nc0 = 0
        self.dx = 6.8e-6  # sensor pixel size (m)
        self.wavelength = 632.8e-9  # HeNe laser wavelength (m)
        self.k = 2 * np.pi / self.wavelength
        self.k_squared = self.k**2
        self.four_pi_squared = 4 * np.pi**2
        self.p = np.pi

        self.input_array = None
        self.output_array = None
        self.fft_object = None
        self.ifft_object = None

        self.x_axis = None
        self.y_axis = None
        self.Fr = None
        self.Fc = None
        self.x = None
        self.y = None
        self.fx = None
        self.fy = None
        self.LPF = None
        self.reference_wave = None

        # Initialize QGraphicsScene and QGraphicsPixmapItem here once
        self.scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.current_title_item = None

        # Create circular progress bar
        self.progress_bar = CircularProgressBar()
        self.progress_bar.hide()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Hologram Reconstruction using Angular Spectrum')
        self.setGeometry(100, 100, 1000, 800)

        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()
        reconstruction_params_layout = QVBoxLayout()

        # --- Hologram Loading & Reconstruction Button Row ---
        button_layout = QHBoxLayout()
        load_button = QPushButton('Load Hologram Image')
        load_button.clicked.connect(self.load_hologram)
        load_button.setStyleSheet("background-color: #1E88E5; color: white; height: 40px; font-size: 14px; border-radius: 8px;font-weight: bold;")

        reconstruct_button = QPushButton('Reconstruct Hologram')
        reconstruct_button.clicked.connect(self.reconstruct_hologram)
        reconstruct_button.setStyleSheet("background-color: #1E88E5; color: white; height: 40px; font-size: 14px; border-radius: 8px; font-weight: bold;")
        button_layout.addSpacing(50)
        button_layout.addWidget(load_button)
        button_layout.addSpacing(100)  # Add horizontal spacing between the buttons
        button_layout.addWidget(reconstruct_button)
        button_layout.addSpacing(50)
        reconstruction_params_layout.addSpacing(20)
        reconstruction_params_layout.addLayout(button_layout)
        reconstruction_params_layout.addSpacing(20)

        # --- Distance controls (existing code) ---
        dist_layout = QHBoxLayout()
        dist_label = QLabel('Reconstruction Distance (m):')
        dist_label.setMinimumHeight(40)
        dist_label.setMaximumHeight(40)
        dist_label.setStyleSheet("font-size: 14px; ")
        self.dist_input = QLineEdit(self)
        self.dist_input.setValidator(QDoubleValidator())
        self.dist_input.setText("0")  # Default value
        self.dist_input.textChanged.connect(self.update_slider_from_text)
        self.dist_input.setMinimumHeight(40)
        self.dist_input.setMaximumHeight(40)
        self.dist_input.setStyleSheet("font-size: 14px;")
        self.dist_slider = QSlider(Qt.Horizontal)
        self.dist_slider.setMinimum(-1000)
        self.dist_slider.setMaximum(1000) # Max 5 meters, scaled
        self.dist_slider.setValue(int(0)) # Default value scaled
        self.dist_slider.setSingleStep(1)
        self.dist_slider.setPageStep(1)   # Optional: Page step (e.g., arrow keys/page up/down)
        self.dist_slider.valueChanged.connect(self.update_text_from_slider)
        self.dist_slider.valueChanged.connect(self.reconstruct_hologram)
        self.dist_slider.setMinimumHeight(40)
        self.dist_slider.setMaximumHeight(40)
        self.dist_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                height: 8px;
                background: #ddd;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50; /* Green */
                border: 1px solid #4CAF50;
                width: 18px;
                height: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #4CAF50; /* Green fill for the left side of the handle */
                border: 1px solid #4CAF50;
                height: 8px;
                border-radius: 4px;
            }
        """)
        dist_layout.addWidget(dist_label)
        dist_layout.addWidget(self.dist_input, 2)
        dist_layout.addSpacing(60)
        dist_layout.addWidget(self.dist_slider, 8)  # QSlider gets 60% of the space
        reconstruction_params_layout.addLayout(dist_layout)
        reconstruction_params_layout.addSpacing(20)
        control_layout.addLayout(reconstruction_params_layout)

        # --- Time Labels ---
        time_labels_layout = QHBoxLayout()
        self.time_label1 = QLabel('Reconstruction Time: N/A')
        self.time_label1.setStyleSheet("font-size: 14px;")
        self.time_label2 = QLabel('Propagation Time: N/A')
        self.time_label2.setStyleSheet("font-size: 14px;")
        self.dimension_label = QLabel('Hologram Dimension: N/A')
        self.dimension_label.setStyleSheet("font-size: 14px;")
        time_labels_layout.addWidget(self.time_label1)
        time_labels_layout.addWidget(self.time_label2)
        time_labels_layout.addWidget(self.dimension_label)
        
        reconstruction_params_layout.addSpacing(20)
        reconstruction_params_layout.addLayout(time_labels_layout)
        reconstruction_params_layout.addSpacing(20)
        main_layout.addLayout(control_layout)

        # --- Create a container for the view and progress bar ---
        view_container = QWidget()
        view_layout = QVBoxLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        
        # QGraphicsView for Display
        self.view = QGraphicsView(self.scene)
        view_layout.addWidget(self.view)
        
        # Add progress bar on top of the view (initially hidden)
        self.progress_bar.setParent(view_container)
        
        main_layout.addWidget(view_container, stretch=7)
        main_layout.setStretchFactor(control_layout, 3)

        # Initial message on the view
        self.set_initial_display_message("   Click on 'Load hologram Image'\n    Select a hologram image\nThen click on 'Reconstruct Hologram'")

        self.setLayout(main_layout)

        # Initialize pyfftw objects
        QTimer.singleShot(100, self.initialize_pyfftw_objects)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Center the progress bar over the view
        if hasattr(self, 'view'):
            view_rect = self.view.geometry()
            progress_x = view_rect.x() + (view_rect.width() - self.progress_bar.width()) // 2
            progress_y = view_rect.y() + (view_rect.height() - self.progress_bar.height()) // 2
            self.progress_bar.move(progress_x, progress_y)

    def set_initial_display_message(self, message):
        # Remove any existing title text item
        if self.current_title_item:
            self.scene.removeItem(self.current_title_item)
            self.current_title_item = None
            
        # Hide the pixmap item if it's showing an old image
        self.pixmap_item.hide()

        # Add a text item
        text_item = QGraphicsTextItem(message)
        font = text_item.font()
        font.setPointSize(12)
        font.setFamily("Arial")
        text_item.setFont(font)
        text_item.setDefaultTextColor(Qt.gray)
        self.scene.addItem(text_item)
        self.current_title_item = text_item
        
        # Center the text in the view
        self.view.setSceneRect(text_item.boundingRect())

    def load_hologram(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Hologram Image", "",
                                                "Image Files (*.bmp *.png *.jpg *.jpeg *.tif *.tiff);;All Files (*)", options=options)
        if file_path:
            # Show progress bar with continuous rotation
            self.show_progress_bar()
            
            # Start loading thread
            self.load_thread = HologramLoadThread(file_path)
            self.load_thread.loading_complete.connect(self.on_hologram_loaded)
            self.load_thread.start()

    def show_progress_bar(self):
        # Center the progress bar over the view
        view_rect = self.view.geometry()
        progress_x = (view_rect.width() - self.progress_bar.width()) // 2
        progress_y = (view_rect.height() - self.progress_bar.height()) // 2
        self.progress_bar.move(progress_x, progress_y)
        self.progress_bar.show()
        self.progress_bar.start_animation()  # Start continuous rotation

    def hide_progress_bar(self):
        self.progress_bar.stop_animation()  # Stop rotation
        self.progress_bar.hide()

    def on_hologram_loaded(self, hologram_data, error_message):
        if error_message:
            self.hide_progress_bar()
            QMessageBox.critical(self, "Error", f"Could not load hologram: {error_message}")
            self.hologram = None
            self.Nr0 = 0
            self.Nc0 = 0
            return

        try:
            self.hologram = hologram_data
            self.Nr0, self.Nc0 = np.shape(self.hologram)
            self.dimension_label.setText(f"Hologram Dimension: {self.Nr0}x {self.Nc0}")

            self.initialize_pyfftw_objects()
            self.precompute_grids()
            
            # Precompute reference wave R after grids are ready
            self.reference_wave = np.exp(-1j * self.k * np.sin(0.04 * self.x))
            self.reference_wave = self.reference_wave / np.max(np.abs(self.reference_wave))
            
            # Generate preview
            self.plot_hologram_preview()
            
            # Final processing
            self.hologram = self.hologram - np.mean(self.hologram)
            self.hologram = self.hologram * self.reference_wave
            
            # Hide progress bar after everything is complete
            self.hide_progress_bar()
            
        except Exception as e:
            self.hide_progress_bar()
            QMessageBox.critical(self, "Error", f"Could not process hologram: {e}")
            self.hologram = None
            self.Nr0 = 0
            self.Nc0 = 0

    def plot_hologram_preview(self):
        if self.hologram is not None and self.hologram.size > 0:
            # Remove any existing title text item
            if self.current_title_item:
                self.scene.removeItem(self.current_title_item)
                self.current_title_item = None

            # Process image for display
            display_image = self.hologram - np.min(self.hologram)
            display_image = (display_image / np.max(display_image) * 255).astype(np.uint8)

            h, w = display_image.shape
            q_image = QImage(display_image.data, w, h, w, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_image)
            self.pixmap_item.setPixmap(pixmap)
            self.pixmap_item.show()
            
            self.view.setSceneRect(self.pixmap_item.boundingRect())
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        else:
            self.set_initial_display_message("No hologram to preview.\nLoad an image.")

    def initialize_pyfftw_objects(self):
        if self.hologram is not None and self.Nr0 > 0 and self.Nc0 > 0:
            try:
                if (self.input_array is None or self.input_array.shape != self.hologram.shape or
                    self.output_array is None or self.output_array.shape != self.hologram.shape):
                    self.input_array = pyfftw.empty_aligned(self.hologram.shape, dtype='complex64')
                    self.output_array = pyfftw.empty_aligned(self.hologram.shape, dtype='complex64')
                    self.fft_object = pyfftw.FFTW(self.input_array, self.output_array, axes=(0, 1),
                                                  direction='FFTW_FORWARD', threads=os.cpu_count())
                    self.ifft_object = pyfftw.FFTW(self.output_array, self.input_array, axes=(0, 1),
                                                   direction='FFTW_BACKWARD', threads=os.cpu_count())
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize pyFFTW: {e}")
                self.fft_object = None
                self.ifft_object = None

    def precompute_grids(self):
        # Use float32 instead of float64 for faster calculations
        self.x_axis = np.linspace(0, self.Nr0 - 1, self.Nr0, dtype=np.float32) - self.Nr0 / 2
        self.y_axis = np.linspace(0, self.Nc0 - 1, self.Nc0, dtype=np.float32) - self.Nc0 / 2
        self.Fr = np.linspace(-0.5, 0.5 - 1/self.Nr0, self.Nr0, dtype=np.float32)
        self.Fc = np.linspace(-0.5, 0.5 - 1/self.Nc0, self.Nc0, dtype=np.float32)

        self.x, self.y = np.meshgrid(self.x_axis, self.y_axis)
        self.fx, self.fy = np.meshgrid(self.Fc, self.Fr)
        self.x = (self.x * self.dx).astype(np.float32)
        self.y = (self.y * self.dx).astype(np.float32)
        self.fx = (self.fx / self.dx).astype(np.float32)
        self.fy = (self.fy / self.dx).astype(np.float32)

    def low_pass_filter(self):
        d2_current = float(self.dist_input.text())
        self.LPF = np.zeros(self.hologram.shape)
        f0 = (1/self.wavelength) * 1/np.sqrt(1 + (4 * d2_current**2 / ((self.Nc0 * self.dx) * (self.Nr0 * self.dx))))
        self.LPF[self.fx**2 + self.fy**2 <= f0**2] = 1
        return self.LPF

    def calculate_asm_parameters(self, d2_val):
        d_orig = 1.054
        f_lens = d_orig/2
        if f_lens == 0:
            L = np.ones(self.hologram.shape, dtype='float64')
        else:
            L = ne.evaluate("exp(1j * p / (f_lens * wavelength) * (x * x + y * y))",
                            local_dict={'p': self.p, 'f_lens': f_lens, 'wavelength': self.wavelength,
                                        'x': self.x, 'y': self.y})

        alpha_squared = ne.evaluate("k_squared - four_pi_squared * (fx**2 + fy**2)",
                                      local_dict={'k_squared': self.k_squared, 'four_pi_squared': self.four_pi_squared,
                                                  'fx': self.fx, 'fy': self.fy})
        alpha = np.sqrt(np.maximum(alpha_squared, 0))
        G = ne.evaluate("exp(1j * alpha * d2_val)",
                        local_dict={'alpha': alpha, 'd2_val': d2_val})

        if self.LPF is not None and np.sum(self.LPF) != 0:
            G = ne.evaluate("G * LPF", local_dict={'G':G, 'LPF':self.LPF})

        return L, G

    def angular_spectrum_method(self, initial_field, G):
        try:
            initial_field = pyfftw.interfaces.numpy_fft.ifftshift(initial_field.astype('complex64'))
            self.input_array[:] = initial_field
            self.fft_object()
            self.output_array[:] = pyfftw.interfaces.numpy_fft.fftshift(self.output_array)
            self.output_array[:] = self.output_array * G
            self.output_array[:] = pyfftw.interfaces.numpy_fft.ifftshift(self.output_array)
            self.ifft_object()
            ftr = pyfftw.interfaces.numpy_fft.fftshift(self.input_array)
            return ftr
        except Exception as e:
            QMessageBox.critical(self, "Angular Spectrum Error", f"An error occurred during ASM: {e}")
            return None

    def reconstruct_hologram(self):
        start_total = time.perf_counter()
        if self.hologram is None or self.hologram.size == 0:
            QMessageBox.warning(self, "No Hologram", "Please load a hologram image first.")
        
        d2 = float(self.dist_input.text())
        L, G = self.calculate_asm_parameters(d2)
        start_time_reconstruction = time.perf_counter()
        ftr = self.angular_spectrum_method(self.hologram, G)
        end_time_reconstruction = time.perf_counter()

        if ftr is not None:
            # Remove any existing title text item
            if self.current_title_item:
                self.scene.removeItem(self.current_title_item)
                self.current_title_item = None

            # Convert the reconstructed intensity to a displayable 8-bit grayscale image
            intensity = np.abs(ftr)**2
            
            min_val = np.min(intensity)
            max_val = np.max(intensity)
            
            if max_val - min_val > 1e-9:
                display_image = (intensity - min_val) / (max_val - min_val) * 255
            else:
                display_image = np.zeros_like(intensity)
            
            display_image = display_image.astype(np.uint8)

            h, w = display_image.shape
            q_image = QImage(display_image.data, w, h, w, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_image)
            self.pixmap_item.setPixmap(pixmap)
            self.pixmap_item.show()

            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self.view.setRenderHint(QPainter.SmoothPixmapTransform)

        self.time_label2.setText(f"Propagation Time: {end_time_reconstruction - start_time_reconstruction:.4f} s")
        end_total = time.perf_counter()
        self.time_label1.setText(f"Total Reconstruction Time: {end_total - start_total:.4f} s")

    def update_slider_from_text(self):
        try:
            value = float(self.dist_input.text())
            slider_value = int(value * 1000)
            if -1000 <= slider_value <= 1000:
                self.dist_slider.setValue(slider_value)
            elif slider_value < -1000:
                self.dist_slider.setValue(-1000)
            else:
                self.dist_slider.setValue(1000)
        except ValueError:
            pass

    def update_text_from_slider(self):
        value = self.dist_slider.value() / 1000.0
        self.dist_input.setText(f"{value:.3f}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet("QWidget { font-family: 'Arial' }")
    ex = HologramReconstructionApp()
    ex.show()
    sys.exit(app.exec_())