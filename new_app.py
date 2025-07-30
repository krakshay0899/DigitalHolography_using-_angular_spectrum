import sys
import os
import numpy as np
import pyfftw
import numexpr as ne
import time
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from PyQt5.QtGui import QDoubleValidator, QImage, QPixmap, QPainter, QPen, QConicalGradient
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSlider, QFileDialog, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem,
    QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QRectF
from PyQt5.QtGui import QColor, QFont
import matplotlib.pyplot as plt


# --- Widgets from holo_gui.py ---
class CircularProgressBar(QWidget):
    """
    A custom circular progress bar widget with a continuous rotation animation.
    This is used to indicate background processing in the application.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 100)
        self.setStyleSheet("background: transparent;")
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_animation)
        self.angle_offset = 0
        # Start the animation timer, updating every 30ms (~33 FPS)
        self.animation_timer.start(30)

    def start_animation(self):
        """Starts the continuous rotation animation."""
        self.animation_timer.start(30)

    def stop_animation(self):
        """Stops the animation."""
        self.animation_timer.stop()

    def update_animation(self):
        """Updates the angle offset for the animation and triggers a repaint."""
        self.angle_offset = (self.angle_offset + 8) % 360  # Increased rotation speed
        self.update()

    def paintEvent(self, event):
        """
        Paints the circular progress bar.
        Draws a background circle and an animated rotating arc with a gradient.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw background circle
        pen_bg = QPen(QColor("#9EBCD6"), 6, Qt.SolidLine)
        painter.setPen(pen_bg)
        painter.drawEllipse(15, 15, 70, 70)

        # Draw animated rotating arc
        rect = QRectF(15, 15, 70, 70)
        start_angle = self.angle_offset * 16  # Start angle rotates (PyQt angles are 1/16th of a degree)
        span_angle = 120 * 16  # Fixed arc length (120 degrees)

        # Create a conical gradient for the rotating arc
        gradient = QConicalGradient(rect.center(), self.angle_offset)
        gradient.setColorAt(0.0, QColor("#5D8EB9"))
        gradient.setColorAt(0.5, QColor("#1969AE"))  # Cyan-like color
        gradient.setColorAt(1.0, QColor("#5D8EB9"))

        pen_progress = QPen()
        pen_progress.setWidth(6)
        pen_progress.setBrush(gradient)
        painter.setPen(pen_progress)
        painter.drawArc(rect, start_angle, span_angle)


class HologramLoadThread(QThread):
    """
    A QThread subclass for loading hologram images in a separate thread.
    This prevents the GUI from freezing during image loading.
    """
    # Signal emitted upon completion: hologram_data (numpy array), error_message (str)
    loading_complete = pyqtSignal(object, str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        """
        Loads the image from the specified file path.
        Converts it to grayscale if it's a color image and ensures it's float type.
        Emits loading_complete signal with data or error message.
        """
        try:
            # Load the hologram using matplotlib's imread (supports various formats)
            hologram_loaded = plt.imread(self.file_path)

            # Convert to grayscale if it's a color image (3D array)
            if hologram_loaded.ndim == 3:
                hologram_loaded = np.mean(hologram_loaded, axis=2)

            # Ensure data type is float for calculations
            hologram_data = hologram_loaded.astype(float)

            if hologram_data.size == 0:
                raise ValueError("Loaded hologram is empty.")

            # Small delay to show loading animation for better UX
            self.msleep(500)

            self.loading_complete.emit(hologram_data, "")

        except Exception as e:
            # Emit error message if loading fails
            self.loading_complete.emit(None, str(e))


class HologramGeneratorThread(QThread):
    """
    A QThread subclass for performing hologram generation and reconstruction
    in a separate thread to keep the GUI responsive.
    """
    # Signals emitted upon completion
    generation_complete = pyqtSignal(object, object, str)  # hologram, fft_of_hologram, error
    # Removed reconstruction_complete signal as it's no longer needed in GenerationTab

    def __init__(self, img1, img2, wavelength, dx, theta, prop_dist, del_z):
        super().__init__()
        self.img1 = img1
        self.img2 = img2
        self.wavelength = wavelength
        self.dx = dx
        self.theta = theta
        self.prop_dist = prop_dist
        self.del_z = del_z
        self.hologram = None # Store generated hologram for reconstruction

    def run(self):
        """
        Executes the hologram generation process.
        If successful, it stores the generated hologram and emits the generation_complete signal.
        """
        try:
            hologram, fft_hologram = self.generate_hologram_from_images()
            if hologram is None:
                raise ValueError("Hologram generation failed.")
            self.hologram = hologram # Store the generated hologram
            self.generation_complete.emit(hologram, fft_hologram, "")

        except Exception as e:
            self.generation_complete.emit(None, None, str(e))

    def generate_hologram_from_images(self):
        """
        Generates a hologram from two input images using the angular spectrum method.
        This function is adapted from the original holo_generator.py.
        """
        u1 = self.img1.astype(np.float32)
        u2 = self.img2.astype(np.float32)

        # Normalize images
        u1 = u1 / np.sum(u1)
        u2 = u2 / np.sum(u2)

        # Zero padding
        p = 0.5  # Padding percentage
        pad_y = int(p * u1.shape[0])
        pad_x = int(p * u1.shape[1])
        u1_padded = np.pad(u1, ((pad_y, pad_y), (pad_x, pad_x)), mode='edge')
        u2_padded = np.pad(u2, ((pad_y, pad_y), (pad_x, pad_x)), mode='edge')
        
        ny, nx = u1_padded.shape
        k = 2 * np.pi / self.wavelength
        
        # Frequency grids
        fx = np.linspace(-0.5, 0.5 - (1/nx), nx) / self.dx
        fy = np.linspace(-0.5, 0.5 - (1/ny), ny) / self.dx
        FX, FY = np.meshgrid(fx, fy)

        def angular_spectrum_propagation(u0, z):
            """Performs angular spectrum propagation."""
            # Low-pass filter based on the provided formula
            f0 = (1 / self.wavelength) * 1 / np.sqrt(1 + (4 * z ** 2 / ((nx * self.dx) * (ny * self.dx))))
            LPF = np.zeros((ny, nx))
            LPF[FX**2 + FY**2 <= f0**2] = 1

            H = np.exp(1j * z * np.sqrt(k**2 - 4 * np.pi**2 * (FX**2 + FY**2)))
            U0 = fftshift(fft2(ifftshift(u0)))
            Uz = U0 * H * LPF
            uz = fftshift(ifft2(ifftshift(Uz)))
            # Remove padding
            return uz[pad_y:-pad_y, pad_x:-pad_x]

        # Propagate the second image
        u2_0 = angular_spectrum_propagation(u2_padded, self.del_z)
        u2_0 = u2_0 / np.sum(u2_0)  # Normalize
        u2_0_padded = np.pad(u2_0, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=0)

        # Propagate combined field
        uz = angular_spectrum_propagation(u1_padded + u2_0_padded, self.prop_dist)
        
        # Create spatial coordinates for reference wave
        x = np.linspace(0, nx-1, nx) - nx/2
        x = (x * self.dx)[pad_x:-pad_x] # Match dimension of uz

        # Reference wave (plane wave with angle theta)
        R = np.exp(1j * k * np.sin(self.theta * x))
        R = R / np.max(np.abs(R)) # Normalize reference wave
        
        # Pad uz to match R's dimensions for element-wise operation (if needed, ensure shapes align)
        # For this specific case, R is 1D and uz is 2D. We need to broadcast R correctly.
        # Assuming R is meant to be applied across the x-dimension of uz.
        # Reshape R to (1, nx_unpadded) and let numpy broadcast.
        R_2D = np.tile(R, (uz.shape[0], 1))

        # Hologram generation
        hologram = np.abs(uz + R_2D)**2

        # Calculate FFT of hologram for display (DC suppression)
        h_fft = fftshift(fft2(ifftshift(hologram)))
        h_fft[int(h_fft.shape[0] // 2), int(h_fft.shape[1] // 2)] = 0  # Set central pixel to 0

        return hologram, np.abs(h_fft)

    # Removed reconstruct method as it's no longer needed in GenerationTab
    # def reconstruct(self, hologram_data, recon_dist):
    #     """
    #     Reconstructs an image from a given hologram.
    #     This function is adapted from the original holo_generator.py.
    #     """
    #     try:
    #         hologram = hologram_data - np.mean(hologram_data)  # DC suppression
    #         h, w = hologram.shape
            
    #         # Padding for reconstruction
    #         p = 0.5
    #         pad_y = int(p * h)
    #         pad_x = int(p * w)
    #         reco_padded = np.pad(hologram, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=0)

    #         ny, nx = reco_padded.shape
    #         k = 2 * np.pi / self.wavelength
            
    #         # Frequency grids for reconstruction
    #         fx = np.linspace(-0.5, 0.5 - (1/nx), nx) / self.dx
    #         fy = np.linspace(-0.5, 0.5 - (1/ny), ny) / self.dx
    #         FX, FY = np.meshgrid(fx, fy)

    #         def angular_spectrum_propagation(u0, z):
    #             """Performs angular spectrum propagation for reconstruction."""
    #             f0 = (1 / self.wavelength) * 1 / np.sqrt(1 + (4 * z ** 2 / ((nx * self.dx) * (ny * self.dx))))
    #             LPF = np.zeros((ny, nx))
    #             LPF[FX**2 + FY**2 <= f0**2] = 1

    #             H = np.exp(1j * z * np.sqrt(k**2 - 4 * np.pi**2 * (FX**2 + FY**2)))
    #             U0 = fftshift(fft2(ifftshift(u0)))
    #             Uz = U0 * H * LPF
    #             uz = fftshift(ifft2(ifftshift(Uz)))
    #             # Remove padding
    #             return uz[pad_y:-pad_y, pad_x:-pad_x]
            
    #         # Spatial coordinates for conjugate reference wave
    #         x = np.linspace(0, nx - 1, nx) - nx / 2
    #         x = (x * self.dx)[pad_x:-pad_x]

    #         # Conjugate of reference wave
    #         R_conj = np.exp(-1j * k * np.sin(self.theta * x))
    #         R_conj = R_conj / np.max(np.abs(R_conj))
    #         R_conj_2D = np.tile(R_conj, (hologram.shape[0], 1)) # Broadcast to 2D

    #         # Multiply hologram with conjugate of reference wave
    #         reconstructed_field_initial = hologram * R_conj_2D
            
    #         # Pad the initial reconstructed field for propagation
    #         reconstructed_field_initial_padded = np.pad(reconstructed_field_initial, ((pad_y, pad_y), (pad_x, pad_x)), constant_values=0)

    #         # Propagate to reconstruction distance
    #         recon_field = angular_spectrum_propagation(reconstructed_field_initial_padded, recon_dist)

    #         self.reconstruction_complete.emit(np.abs(recon_field)**2, "")
    #     except Exception as e:
    #         self.reconstruction_complete.emit(None, str(e))


# --- Renamed and Modified Tabs ---
class ReconstructionTab(QWidget):
    """
    This tab handles the reconstruction of a hologram.
    It provides controls for loading a hologram, adjusting reconstruction distance,
    wavelength, and angle, and displays the reconstructed image.
    """
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
        self.theta = 1  # Angle in degrees for reference wave

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

        # Initialize QGraphicsScene and QGraphicsPixmapItem for display
        self.scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.current_title_item = None # To manage initial message text item

        # Create circular progress bar for loading/processing indication
        self.progress_bar = CircularProgressBar()
        self.progress_bar.hide() # Initially hidden

        self.initUI()
        
    def initUI(self):
        """Initializes the user interface for the Reconstruction Tab."""
        main_layout = QVBoxLayout(self)
        control_layout = QHBoxLayout()
        reconstruction_params_layout = QVBoxLayout()

        # --- Hologram Loading & Dimension Label Row ---
        button_layout = QHBoxLayout()
        load_button = QPushButton('Load Hologram Image')
        load_button.clicked.connect(self.load_hologram)
        load_button.setStyleSheet("background-color: #1E88E5; color: white; height: 40px; font-size: 14px; border-radius: 8px;font-weight: bold;")

        self.dimension_label = QLabel('Hologram Dimension: N/A')
        self.dimension_label.setStyleSheet("font-size: 14px;font-weight: bold;")
        self.dimension_label.setAlignment(Qt.AlignCenter)
        button_layout.addSpacing(50)
        button_layout.addWidget(load_button)
        button_layout.addSpacing(100)
        button_layout.addWidget(self.dimension_label)
        button_layout.addSpacing(50)
        reconstruction_params_layout.addSpacing(20)
        reconstruction_params_layout.addLayout(button_layout)
        reconstruction_params_layout.addSpacing(20)

        # --- Distance controls ---
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
        self.dist_input.setStyleSheet("font-size: 14px; border: 2px solid #ccc; border-radius: 8px; padding: 5px;")
        self.dist_slider = QSlider(Qt.Horizontal)
        self.dist_slider.setMinimum(-1000) # Scaled for 0.001m increments
        self.dist_slider.setMaximum(1000) # Max 1.000 meters
        self.dist_slider.setValue(int(0)) # Default value scaled
        self.dist_slider.setSingleStep(1)
        self.dist_slider.setPageStep(100) # Page step for larger jumps
        self.dist_slider.valueChanged.connect(self.update_text_from_slider)
        self.dist_slider.valueChanged.connect(self.reconstruct_hologram) # Trigger reconstruction on slider change
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
        dist_layout.addWidget(self.dist_slider, 8)
        reconstruction_params_layout.addLayout(dist_layout)
        reconstruction_params_layout.addSpacing(20)

        # --- Parameters controls (Wavelength, Angle, Lens Focal Length) ---
        params_layout = QHBoxLayout()
        
        # Wavelength controls
        wavelength_label = QLabel('Wavelength (nm):')
        wavelength_label.setMinimumHeight(40)
        wavelength_label.setMaximumHeight(40)
        wavelength_label.setStyleSheet("font-size: 14px; ")
        self.wavelength_input = QLineEdit(self)
        self.wavelength_input.setValidator(QDoubleValidator())
        self.wavelength_input.setText("632")  # Default value 632 nm
        self.wavelength_input.textChanged.connect(self.update_wavelength)
        self.wavelength_input.setMinimumHeight(40)
        self.wavelength_input.setMaximumHeight(40)
        self.wavelength_input.setStyleSheet("font-size: 14px; border: 2px solid #ccc; border-radius: 8px; padding: 5px;")
        params_layout.addWidget(wavelength_label)
        params_layout.addWidget(self.wavelength_input)
        params_layout.addSpacing(20)

        # Angle controls
        angle_label = QLabel('Angle(deg.):')
        angle_label.setMinimumHeight(40)
        angle_label.setMaximumHeight(40)
        angle_label.setStyleSheet("font-size: 14px; ")
        self.angle_input = QLineEdit(self)
        self.angle_input.setValidator(QDoubleValidator())
        self.angle_input.setText("1")  # Default value
        self.angle_input.textChanged.connect(self.update_angle)
        self.angle_input.setMinimumHeight(40)
        self.angle_input.setMaximumHeight(40)
        self.angle_input.setStyleSheet("font-size: 14px; border: 2px solid #ccc; border-radius: 8px; padding: 5px;")
        params_layout.addWidget(angle_label)
        params_layout.addWidget(self.angle_input)
        params_layout.addSpacing(20)

        # Lens focal length controls (currently not used in calculation, but kept for UI)
        focal_length_label = QLabel('Lens Focal Length (cm):')
        focal_length_label.setMinimumHeight(40)
        focal_length_label.setMaximumHeight(40)
        focal_length_label.setStyleSheet("font-size: 14px; ")
        self.focal_length_input = QLineEdit(self)
        self.focal_length_input.setValidator(QDoubleValidator())
        self.focal_length_input.setText("30")  # Default value
        self.focal_length_input.setMinimumHeight(40)
        self.focal_length_input.setMaximumHeight(40)
        self.focal_length_input.setStyleSheet("font-size: 14px; border: 2px solid #ccc; border-radius: 8px; padding: 5px;")
        params_layout.addWidget(focal_length_label)
        params_layout.addWidget(self.focal_length_input)

        reconstruction_params_layout.addLayout(params_layout)
        reconstruction_params_layout.addSpacing(20)
        
        control_layout.addLayout(reconstruction_params_layout)

        # --- Time Labels & Reconstruct Button ---
        time_labels_layout = QHBoxLayout()
        self.time_label1 = QLabel('Reconstruction Time: N/A')
        self.time_label1.setStyleSheet("font-size: 14px;")
        self.time_label1.setAlignment(Qt.AlignCenter)
        reconstruct_button = QPushButton('Reconstruct Hologram')
        reconstruct_button.clicked.connect(self.reconstruct_hologram)
        reconstruct_button.setStyleSheet("background-color: #1E88E5; color: white; height: 40px; font-size: 14px; border-radius: 8px; font-weight: bold;")
        time_labels_layout.addSpacing(50)
        time_labels_layout.addWidget(self.time_label1)
        time_labels_layout.addSpacing(100)
        time_labels_layout.addWidget(reconstruct_button)
        time_labels_layout.addSpacing(50)
        
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
        
        # Configure view properties for proper centering and bounds
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        
        view_layout.addWidget(self.view)
        
        # Add progress bar on top of the view (initially hidden)
        self.progress_bar.setParent(view_container)
        
        main_layout.addWidget(view_container, stretch=7)
        main_layout.setStretchFactor(control_layout, 3)

        # Initial message on the view
        self.set_initial_display_message("   Click on 'Load hologram Image'\n    Select a hologram image\nThen click on 'Reconstruct Hologram'")

        self.setLayout(main_layout)

        # Initialize pyfftw objects after UI is set up
        QTimer.singleShot(100, self.initialize_pyfftw_objects)

    def resizeEvent(self, event):
        """Overrides resize event to center the progress bar."""
        super().resizeEvent(event)
        # Center the progress bar over the view
        if hasattr(self, 'view'):
            view_rect = self.view.geometry()
            progress_x = view_rect.x() + (view_rect.width() - self.progress_bar.width()) // 2
            progress_y = view_rect.y() + (view_rect.height() - self.progress_bar.height()) // 2
            self.progress_bar.move(progress_x, progress_y)

    def set_initial_display_message(self, message):
        """
        Sets an initial message to be displayed in the QGraphicsView.
        This is useful when no image is loaded.
        """
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
        """
        Opens a file dialog to select a hologram image.
        Starts a HologramLoadThread to load the image in the background.
        """
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
        """Shows and starts the animation of the circular progress bar."""
        # Center the progress bar over the view
        view_rect = self.view.geometry()
        progress_x = (view_rect.width() - self.progress_bar.width()) // 2
        progress_y = (view_rect.height() - self.progress_bar.height()) // 2
        self.progress_bar.move(progress_x, progress_y)
        self.progress_bar.show()
        self.progress_bar.start_animation()

    def hide_progress_bar(self):
        """Hides and stops the animation of the circular progress bar."""
        self.progress_bar.stop_animation()
        self.progress_bar.hide()

    def on_hologram_loaded(self, hologram_data, error_message):
        """
        Callback function executed when the hologram loading thread completes.
        Handles loaded data or displays error messages.
        """
        if error_message:
            self.hide_progress_bar()
            QMessageBox.critical(self, "Error", f"Could not load hologram: {error_message}")
            self.hologram = None
            self.Nr0 = 0
            self.Nc0 = 0
            return

        try:
            self.hologram = hologram_data
            self.original_hologram = hologram_data.copy()  # Store original for angle changes
            self.Nr0, self.Nc0 = np.shape(self.hologram)
            self.dimension_label.setText(f"Hologram Dimension: {self.Nr0}x {self.Nc0}")

            self.initialize_pyfftw_objects()
            self.precompute_grids()
            
            # Precompute reference wave R after grids are ready
            # Convert angle from degrees to radians for numpy.sin
            self.reference_wave = np.exp(-1j * self.k * np.sin(self.theta * np.pi / 180 * self.x))
            self.reference_wave = self.reference_wave / np.max(np.abs(self.reference_wave))
            
            # Generate preview of the loaded hologram
            self.plot_hologram_preview()
            
            # Apply DC suppression and reference wave multiplication for reconstruction
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
        """Displays the loaded hologram in the QGraphicsView."""
        if self.hologram is not None and self.hologram.size > 0:
            # Remove any existing title text item
            if self.current_title_item:
                self.scene.removeItem(self.current_title_item)
                self.current_title_item = None

            # Process image for display (normalize to 0-255 uint8)
            display_image = self.hologram - np.min(self.hologram)
            if np.max(display_image) > 0:
                display_image = (display_image / np.max(display_image) * 255).astype(np.uint8)
            else:
                display_image = np.zeros_like(self.hologram, dtype=np.uint8) # Handle case of all zero/constant image

            h, w = display_image.shape
            q_image = QImage(display_image.data, w, h, w, QImage.Format_Grayscale8)

            pixmap = QPixmap.fromImage(q_image)
            self.pixmap_item.setPixmap(pixmap)
            self.pixmap_item.show()
            
            # Reset view transformation to ensure clean state
            self.view.resetTransform()
            
            # Set scene rect with margin for better centering
            pixmap_rect = self.pixmap_item.boundingRect()
            margin = min(pixmap_rect.width(), pixmap_rect.height()) * 0.1  # 10% margin
            expanded_rect = pixmap_rect.adjusted(-margin, -margin, margin, margin)
            self.scene.setSceneRect(expanded_rect)
            
            # Center and fit the image
            self.view.centerOn(self.pixmap_item)
            self.view.fitInView(pixmap_rect, Qt.KeepAspectRatio)
            self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        else:
            self.set_initial_display_message("No hologram to preview.\nLoad an image.")

    def initialize_pyfftw_objects(self):
        """
        Initializes pyFFTW planning objects for optimized FFT/IFFT.
        This is done once after hologram dimensions are known.
        """
        if self.hologram is not None and self.Nr0 > 0 and self.Nc0 > 0:
            try:
                # Re-initialize only if hologram shape changes
                if (self.input_array is None or self.input_array.shape != self.hologram.shape or
                    self.output_array is None or self.output_array.shape != self.hologram.shape):
                    self.input_array = pyfftw.empty_aligned(self.hologram.shape, dtype='complex64')
                    self.output_array = pyfftw.empty_aligned(self.hologram.shape, dtype='complex64')
                    # Plan FFT and IFFT
                    self.fft_object = pyfftw.FFTW(self.input_array, self.output_array, axes=(0, 1),
                                                  direction='FFTW_FORWARD', threads=os.cpu_count())
                    self.ifft_object = pyfftw.FFTW(self.output_array, self.input_array, axes=(0, 1),
                                                   direction='FFTW_BACKWARD', threads=os.cpu_count())
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize pyFFTW: {e}")
                self.fft_object = None
                self.ifft_object = None

    def precompute_grids(self):
        """
        Precomputes spatial and frequency grids used in propagation calculations.
        Uses float32 for potentially faster computations.
        """
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
        """Calculates and returns a low-pass filter for the current reconstruction distance."""
        d2_current = float(self.dist_input.text())
        self.LPF = np.zeros(self.hologram.shape)
        # Formula for cutoff frequency f0
        f0 = (1/self.wavelength) * 1/np.sqrt(1 + (4 * d2_current**2 / ((self.Nc0 * self.dx) * (self.Nr0 * self.dx))))
        self.LPF[self.fx**2 + self.fy**2 <= f0**2] = 1
        return self.LPF

    def calculate_asm_parameters(self, d2_val):
        """
        Calculates the parameters (L and G) for the Angular Spectrum Method (ASM).
        L is for lens phase, G is the propagation transfer function.
        """
        # d_orig and f_lens seem to be fixed values for a specific setup, adjust if they are meant to be dynamic
        d_orig = 1.054
        f_lens = d_orig/2
        if f_lens == 0: # Avoid division by zero
            L = np.ones(self.hologram.shape, dtype='float64')
        else:
            # Using numexpr for potentially faster element-wise operations
            L = ne.evaluate("exp(1j * p / (f_lens * wavelength) * (x * x + y * y))",
                            local_dict={'p': self.p, 'f_lens': f_lens, 'wavelength': self.wavelength,
                                        'x': self.x, 'y': self.y})

        alpha_squared = ne.evaluate("k_squared - four_pi_squared * (fx**2 + fy**2)",
                                      local_dict={'k_squared': self.k_squared, 'four_pi_squared': self.four_pi_squared,
                                                  'fx': self.fx, 'fy': self.fy})
        alpha = np.sqrt(np.maximum(alpha_squared, 0)) # Ensure non-negative argument for sqrt
        G = ne.evaluate("exp(1j * alpha * d2_val)",
                        local_dict={'alpha': alpha, 'd2_val': d2_val})

        # Apply low-pass filter if it exists and is not all zeros
        if self.LPF is not None and np.sum(self.LPF) != 0:
            G = ne.evaluate("G * LPF", local_dict={'G':G, 'LPF':self.LPF})

        return L, G

    def angular_spectrum_method(self, initial_field, G):
        """
        Performs the Angular Spectrum Method propagation using pyFFTW.
        Handles FFT, multiplication with transfer function, and IFFT.
        """
        try:
            # Shift input for FFT, copy to aligned array
            initial_field_shifted = pyfftw.interfaces.numpy_fft.ifftshift(initial_field.astype('complex64'))
            self.input_array[:] = initial_field_shifted
            
            # Execute FFT
            self.fft_object()
            
            # Shift back, multiply by transfer function, then shift for IFFT
            self.output_array[:] = pyfftw.interfaces.numpy_fft.fftshift(self.output_array)
            self.output_array[:] = self.output_array * G
            self.output_array[:] = pyfftw.interfaces.numpy_fft.ifftshift(self.output_array)
            
            # Execute IFFT
            self.ifft_object()
            
            # Final shift for result
            ftr = pyfftw.interfaces.numpy_fft.fftshift(self.input_array)
            return ftr
        except Exception as e:
            QMessageBox.critical(self, "Angular Spectrum Error", f"An error occurred during ASM: {e}")
            return None

    def reconstruct_hologram(self):
        """
        Triggers the hologram reconstruction process based on current parameters.
        Updates the displayed image and reconstruction time.
        """
        start_total = time.perf_counter()
        if self.hologram is None or self.hologram.size == 0:
            QMessageBox.warning(self, "No Hologram", "Please load a hologram image first.")
            return # Exit if no hologram is loaded
        
        try:
            d2 = float(self.dist_input.text())
            # Recalculate LPF if distance changes
            self.low_pass_filter()
            L, G = self.calculate_asm_parameters(d2)
            
            ftr = self.angular_spectrum_method(self.hologram, G)

            if ftr is not None:
                # Remove any existing title text item
                if self.current_title_item:
                    self.scene.removeItem(self.current_title_item)
                    self.current_title_item = None

                # Convert the reconstructed intensity to a displayable 8-bit grayscale image
                intensity = np.abs(ftr)**2
                
                min_val = np.min(intensity)
                max_val = np.max(intensity)
                
                # Normalize to 0-255 range, handle case where max_val == min_val
                if max_val - min_val > 1e-9: # Check for non-zero range
                    display_image = (intensity - min_val) / (max_val - min_val) * 255
                else:
                    display_image = np.zeros_like(intensity) # All zeros if image is constant
                
                display_image = display_image.astype(np.uint8)

                h, w = display_image.shape
                # Create QImage from numpy array (Grayscale8 format)
                q_image = QImage(display_image.data, w, h, w, QImage.Format_Grayscale8)

                pixmap = QPixmap.fromImage(q_image)
                self.pixmap_item.setPixmap(pixmap)
                self.pixmap_item.show()

                # Reset view transformation to ensure clean state
                self.view.resetTransform()

                # Set scene rect with margin for better centering
                pixmap_rect = self.pixmap_item.boundingRect()
                margin = min(pixmap_rect.width(), pixmap_rect.height()) * 0.1  # 10% margin
                expanded_rect = pixmap_rect.adjusted(-margin, -margin, margin, margin)
                self.scene.setSceneRect(expanded_rect)
                
                # Center and fit the image
                self.view.centerOn(self.pixmap_item)
                self.view.fitInView(pixmap_rect, Qt.KeepAspectRatio)
                self.view.setRenderHint(QPainter.SmoothPixmapTransform) # For better scaling quality
        except Exception as e:
            QMessageBox.critical(self, "Reconstruction Error", f"An error occurred during reconstruction: {e}")

        end_total = time.perf_counter()
        self.time_label1.setText(f"Total Reconstruction Time: {end_total - start_total:.4f} s")

    def update_slider_from_text(self):
        """Updates the slider position based on the text input for distance."""
        try:
            value = float(self.dist_input.text())
            slider_value = int(value * 1000) # Scale to slider range
            # Clamp value to slider's min/max
            if -1000 <= slider_value <= 1000:
                self.dist_slider.setValue(slider_value)
            elif slider_value < -1000:
                self.dist_slider.setValue(-1000)
            else:
                self.dist_slider.setValue(1000)
        except ValueError:
            pass # Ignore invalid text input

    def update_text_from_slider(self):
        """Updates the text input for distance based on the slider position."""
        value = self.dist_slider.value() / 1000.0 # Scale back to meters
        self.dist_input.setText(f"{value:.3f}")

    def update_wavelength(self):
        """Updates the wavelength parameter and triggers reference wave recalculation."""
        try:
            wavelength_nm = float(self.wavelength_input.text())
            self.wavelength = wavelength_nm * 1e-9  # Convert nm to m
            # Update k and k_squared based on new wavelength
            self.k = 2 * np.pi / self.wavelength
            self.k_squared = self.k**2
            # Recalculate reference wave and update hologram if loaded
            if self.hologram is not None and hasattr(self, 'x'):
                self.update_reference_wave()
        except ValueError:
            pass

    def update_angle(self):
        """Updates the angle parameter and triggers reference wave recalculation."""
        try:
            angle_deg = float(self.angle_input.text())
            self.theta = angle_deg  # Store angle in degrees for use in reference wave calculation
            # Recalculate reference wave and update hologram if loaded
            if self.hologram is not None and hasattr(self, 'x'):
                self.update_reference_wave()
        except ValueError:
            pass

    def update_reference_wave(self):
        """
        Recalculates the reference wave and reapplies it to the hologram.
        This is called when wavelength or angle parameters change.
        """
        if self.hologram is not None and hasattr(self, 'x'):
            # Recalculate reference wave with new angle (convert degrees to radians)
            self.reference_wave = np.exp(-1j * self.k * np.sin(self.theta * np.pi / 180 * self.x))
            self.reference_wave = self.reference_wave / np.max(np.abs(self.reference_wave))
            
            # Reapply reference wave to original hologram data
            # We need to use the stored original_hologram to avoid cumulative changes
            if hasattr(self, 'original_hologram'):
                self.hologram = self.original_hologram - np.mean(self.original_hologram)
                self.hologram = self.hologram * self.reference_wave
                
                # Trigger reconstruction if distance slider is connected
                self.reconstruct_hologram()


class GenerationTab(QWidget):
    """
    This tab handles the generation of a hologram from two source images.
    The UI is designed to be similar to the ReconstructionTab for consistency.
    """
    def __init__(self):
        super().__init__()
        self.img1 = None
        self.img2 = None
        self.hologram = None # Stores the generated hologram

        # Default parameters for generation
        self.dx = 6.8e-6
        self.wavelength = 632.8e-9
        self.theta = 0.04
        self.prop_dist = 0.1
        self.del_z = 0.1

        # UI components
        self.scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.current_title_item = None

        self.progress_bar = CircularProgressBar()
        self.progress_bar.hide()
        
        self.initUI()
        
    def initUI(self):
        """Initializes the user interface for the Generation Tab."""
        main_layout = QVBoxLayout(self)
        control_layout = QHBoxLayout()
        generation_params_layout = QVBoxLayout()

        # --- Image Loading Controls ---
        img_controls_layout = QHBoxLayout()
        load_img1_btn = QPushButton("Load Image 1")
        load_img1_btn.clicked.connect(self.load_image_1)
        load_img1_btn.setStyleSheet("background-color: #1E88E5; color: white; height: 40px; font-size: 14px; border-radius: 8px;font-weight: bold;")
        self.img1_label = QLabel("Image 1: N/A")
        self.img1_label.setStyleSheet("font-size: 14px;")
        
        load_img2_btn = QPushButton("Load Image 2")
        load_img2_btn.clicked.connect(self.load_image_2)
        load_img2_btn.setStyleSheet("background-color: #1E88E5; color: white; height: 40px; font-size: 14px; border-radius: 8px;font-weight: bold;")
        self.img2_label = QLabel("Image 2: N/A")
        self.img2_label.setStyleSheet("font-size: 14px;")
        
        img_controls_layout.addSpacing(50)
        img_controls_layout.addWidget(load_img1_btn)
        img_controls_layout.addWidget(self.img1_label)
        img_controls_layout.addSpacing(100)
        img_controls_layout.addWidget(load_img2_btn)
        img_controls_layout.addWidget(self.img2_label)
        img_controls_layout.addSpacing(50)
        generation_params_layout.addLayout(img_controls_layout)
        generation_params_layout.addSpacing(20)

        # --- Generation Parameters ---
        params_layout1 = QHBoxLayout()
        self.wavelength_input = self.create_param_input(params_layout1, "Wavelength (nm):", "632.8")
        self.dx_input = self.create_param_input(params_layout1, "Pixel Size (μm):", "6.8")
        generation_params_layout.addLayout(params_layout1)
        generation_params_layout.addSpacing(10)

        params_layout2 = QHBoxLayout()
        self.prop_dist_input = self.create_param_input(params_layout2, "Prop. Dist. (m):", "0.1")
        self.del_z_input = self.create_param_input(params_layout2, "Delta Z (m):", "0.1")
        self.theta_input = self.create_param_input(params_layout2, "Angle (rad):", "0.04")
        generation_params_layout.addLayout(params_layout2)
        generation_params_layout.addSpacing(20)

        # --- Action Buttons ---
        action_layout = QHBoxLayout()
        generate_btn = QPushButton("Generate Hologram")
        generate_btn.clicked.connect(self.start_hologram_generation)
        generate_btn.setStyleSheet("background-color: #4CAF50; color: white; height: 40px; font-size: 14px; border-radius: 8px;font-weight: bold;")
        
        save_btn = QPushButton("Save Hologram")
        save_btn.clicked.connect(self.save_hologram)
        save_btn.setStyleSheet("background-color: #FFC107; color: black; height: 40px; font-size: 14px; border-radius: 8px;font-weight: bold;")
        
        action_layout.addStretch()
        action_layout.addWidget(generate_btn)
        action_layout.addSpacing(50)
        action_layout.addWidget(save_btn)
        action_layout.addStretch()
        generation_params_layout.addLayout(action_layout)
        generation_params_layout.addSpacing(20)
        
        control_layout.addLayout(generation_params_layout)
        main_layout.addLayout(control_layout)

        # --- Display View ---
        view_container = QWidget()
        view_layout = QVBoxLayout(view_container)
        view_layout.setContentsMargins(0, 0, 0, 0)
        
        self.view = QGraphicsView(self.scene)
        
        # Configure view properties for proper centering and bounds
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setDragMode(QGraphicsView.NoDrag)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.view.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        
        view_layout.addWidget(self.view)
        
        self.progress_bar.setParent(view_container)
        
        main_layout.addWidget(view_container, stretch=7)
        main_layout.setStretchFactor(control_layout, 3)

        self.set_initial_display_message("   Load two images to generate a hologram.\n   Adjust parameters as needed.")
        self.setLayout(main_layout)

    def create_param_input(self, layout, label_text, default_value):
        """Helper function to create a labeled QLineEdit and add it to a layout."""
        param_layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setMinimumHeight(40)
        label.setStyleSheet("font-size: 14px;")
        input_field = QLineEdit(self)
        input_field.setValidator(QDoubleValidator())
        input_field.setText(default_value)
        input_field.setMinimumHeight(40)
        input_field.setStyleSheet("font-size: 14px; border: 2px solid #ccc; border-radius: 8px; padding: 5px;")
        param_layout.addWidget(label)
        param_layout.addWidget(input_field)
        layout.addLayout(param_layout)
        return input_field

    def resizeEvent(self, event):
        """Overrides resize event to center the progress bar."""
        super().resizeEvent(event)
        if hasattr(self, 'view'):
            view_rect = self.view.geometry()
            progress_x = view_rect.x() + (view_rect.width() - self.progress_bar.width()) // 2
            progress_y = view_rect.y() + (view_rect.height() - self.progress_bar.height()) // 2
            self.progress_bar.move(progress_x, progress_y)

    def set_initial_display_message(self, message):
        """Sets an initial message to be displayed in the QGraphicsView."""
        if self.current_title_item:
            self.scene.removeItem(self.current_title_item)
            self.current_title_item = None
        self.pixmap_item.hide()
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
        self.view.centerOn(text_item)

    def load_image_1(self):
        """Loads the first source image for hologram generation."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image 1", "", "Image Files (*.bmp *.png *.jpg *.jpeg *.tif *.tiff)")
        if file_path:
            self.img1 = cv2.imread(file_path, 0) # Load as grayscale
            if self.img1 is None:
                QMessageBox.critical(self, "Error", "Could not load image 1. Please check the file format or path.")
                return
            self.img1_label.setText(f"Image 1: {os.path.basename(file_path)}")

    def load_image_2(self):
        """Loads the second source image for hologram generation."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Image 2", "", "Image Files (*.bmp *.png *.jpg *.jpeg *.tif *.tiff)")
        if file_path:
            self.img2 = cv2.imread(file_path, 0) # Load as grayscale
            if self.img2 is None:
                QMessageBox.critical(self, "Error", "Could not load image 2. Please check the file format or path.")
                return
            self.img2_label.setText(f"Image 2: {os.path.basename(file_path)}")
            
    def get_params(self):
        """Retrieves and validates parameters from input fields."""
        try:
            self.wavelength = float(self.wavelength_input.text()) * 1e-9 # nm to m
            self.prop_dist = float(self.prop_dist_input.text())
            self.del_z = float(self.del_z_input.text())
            self.theta = float(self.theta_input.text()) # radians
            self.dx = float(self.dx_input.text()) * 1e-6 # μm to m
            return True
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Invalid input. Please ensure all parameters are valid numbers.")
            return False

    def start_hologram_generation(self):
        """Initiates the hologram generation process in a separate thread."""
        if self.img1 is None or self.img2 is None:
            QMessageBox.warning(self, "Warning", "Please load both source images first.")
            return
        if not self.get_params(): # Validate parameters before starting
            return
        
        self.show_progress_bar()
        # Create and start the generation thread
        self.generator_thread = HologramGeneratorThread(self.img1, self.img2, self.wavelength, self.dx, self.theta, self.prop_dist, self.del_z)
        self.generator_thread.generation_complete.connect(self.on_hologram_generated)
        self.generator_thread.start()

    def on_hologram_generated(self, hologram_data, fft_hologram, error_message):
        """Callback for when hologram generation thread completes."""
        self.hide_progress_bar()
        if error_message:
            QMessageBox.critical(self, "Generation Error", f"Hologram generation failed: {error_message}")
            return
        self.hologram = hologram_data # Store the generated hologram
        self.display_image(self.scene, self.pixmap_item, hologram_data)

    def display_image(self, scene, pixmap_item, image_data):
        """Helper to display a numpy array as an image in a QGraphicsView."""
        if self.current_title_item:
            self.scene.removeItem(self.current_title_item)
            self.current_title_item = None

        # Normalize image data to 0-255 for display
        display_image = image_data - np.min(image_data)
        if np.max(display_image) > 0:
            display_image = (display_image / np.max(display_image) * 255).astype(np.uint8)
        else:
            display_image = np.zeros_like(image_data, dtype=np.uint8)

        h, w = display_image.shape
        q_image = QImage(display_image.data, w, h, w, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.pixmap_item.setPixmap(pixmap)
        self.pixmap_item.show()
        
        # Reset view transformation to ensure clean state
        self.view.resetTransform()
        
        # Set scene rect with some margin around the pixmap for better centering
        pixmap_rect = self.pixmap_item.boundingRect()
        margin = min(pixmap_rect.width(), pixmap_rect.height()) * 0.1  # 10% margin
        expanded_rect = pixmap_rect.adjusted(-margin, -margin, margin, margin)
        self.scene.setSceneRect(expanded_rect)
        
        # Center the pixmap within the expanded scene
        self.view.centerOn(self.pixmap_item)
        self.view.fitInView(pixmap_rect, Qt.KeepAspectRatio)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)  # For better scaling quality

    def save_hologram(self):
        """Saves the generated hologram to a file."""
        if self.hologram is None:
            QMessageBox.warning(self, "Warning", "No hologram has been generated yet.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Hologram", "", "PNG Files (*.png);;BMP Files (*.bmp);;All Files (*)", options=options)
        if file_path:
            try:
                # Normalize to 0-255 for saving as an image file
                save_image = self.hologram - np.min(self.hologram)
                if np.max(save_image) > 0:
                    save_image = (save_image / np.max(save_image) * 255).astype(np.uint8)
                else:
                    save_image = np.zeros_like(self.hologram, dtype=np.uint8)
                
                cv2.imwrite(file_path, save_image)
                QMessageBox.information(self, "Success", f"Hologram saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save hologram: {e}")

    def show_progress_bar(self):
        """Shows and starts the animation of the circular progress bar."""
        view_rect = self.view.geometry()
        progress_x = (view_rect.width() - self.progress_bar.width()) // 2
        progress_y = (view_rect.height() - self.progress_bar.height()) // 2
        self.progress_bar.move(progress_x, progress_y)
        self.progress_bar.show()
        self.progress_bar.start_animation()

    def hide_progress_bar(self):
        """Hides and stops the animation of the circular progress bar."""
        self.progress_bar.stop_animation()
        self.progress_bar.hide()


class HoloApp(QWidget):
    """
    The main application window that hosts the two tabs:
    Hologram Reconstruction and Hologram Generation.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Holography Application')
        self.setGeometry(100, 100, 1200, 900) # Set initial window size
        
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget() # Create the tab widget
        
        # Instantiate the two tab widgets
        self.reconstruction_tab = ReconstructionTab()
        self.generation_tab = GenerationTab()
        
        # Add tabs to the QTabWidget
        self.tabs.addTab(self.reconstruction_tab, "Hologram Reconstruction")
        self.tabs.addTab(self.generation_tab, "Hologram Generation")
        
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)


if __name__ == '__main__':
    # Ensure pyfftw uses all available CPU cores for optimal performance
    pyfftw.config.threads = os.cpu_count()
    # Set the cache size for pyfftw plans to avoid re-planning frequently
    pyfftw.config.cache_byte_limit = 1 * (2**30) # 1 GB cache

    app = QApplication(sys.argv)
    # Apply a global stylesheet for consistent font
    app.setStyleSheet("QWidget { font-family: 'Arial' }")
    
    # Create and show the main application window
    ex = HoloApp()
    ex.show()
    
    # Start the Qt event loop
    sys.exit(app.exec_())
