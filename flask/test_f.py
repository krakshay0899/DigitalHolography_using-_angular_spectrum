import sys
import os
import numpy as np
import pyfftw
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift
import numexpr as ne
import time
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import uuid # For generating unique filenames

# Ensure pyfftw is configured for optimal performance
pyfftw.config.THREADING = True
pyfftw.config.NUM_THREADS = os.cpu_count()

app = Flask(__name__)
app.secret_key = 'super_secret_key_for_hologram_app'  # Replace with a strong, random key in production
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# --- HologramReconstructionCore (remains mostly the same, but now takes filepath or data directly) ---
class HologramReconstructionCore:
    def __init__(self, hologram_data_np): # Now takes numpy array directly
        self.hologram = hologram_data_np
        self.Nr0, self.Nc0 = self.hologram.shape
        self.dx = 6.8e-6  # sensor pixel size (m)
        self.wavelength = 632.8e-9  # HeNe laser wavelength (m)
        self.k = 2 * np.pi / self.wavelength
        self.k_squared = self.k**2
        self.four_pi_squared = 4 * np.pi**2
        self.p = np.pi

        # Initialize pyfftw objects
        # Ensure that input_array and output_array are correctly sized based on hologram.shape
        self.input_array = pyfftw.empty_aligned(self.hologram.shape, dtype='complex64')
        self.output_array = pyfftw.empty_aligned(self.hologram.shape, dtype='complex64')
        self.fft_object = pyfftw.FFTW(self.input_array, self.output_array, axes=(0, 1),
                                     direction='FFTW_FORWARD', threads=pyfftw.config.NUM_THREADS)
        self.ifft_object = pyfftw.FFTW(self.output_array, self.input_array, axes=(0, 1),
                                      direction='FFTW_BACKWARD', threads=pyfftw.config.NUM_THREADS)

        self.precompute_grids()
        self.reference_wave = np.exp(-1j * self.k * np.sin(0.04 * self.x))
        self.reference_wave = self.reference_wave / np.max(np.abs(self.reference_wave))
        self.hologram = self.hologram - np.mean(self.hologram)  # Remove DC component
        self.hologram = self.hologram * self.reference_wave  # Multiply hologram


    def precompute_grids(self):
        self.x_axis = np.linspace(0, self.Nr0 - 1, self.Nr0) - self.Nr0 / 2
        self.y_axis = np.linspace(0, self.Nc0 - 1, self.Nc0) - self.Nc0 / 2
        self.Fr = np.linspace(-0.5, 0.5 - 1/self.Nr0, self.Nr0)
        self.Fc = np.linspace(-0.5, 0.5 - 1/self.Nc0, self.Nc0)

        # Correct meshgrid order: (columns for x, rows for y) if dx, dy are associated with columns/rows
        # In typical image processing, first dimension is rows (height), second is columns (width)
        # If x corresponds to columns (width) and y to rows (height), then meshgrid order should be careful
        # Assuming your original x, y were for (rows, columns) as in (Nr0, Nc0)
        self.x, self.y = np.meshgrid(self.y_axis, self.x_axis) # Corrected to match image (Nc0 x Nr0) if x is horizontal, y is vertical
        self.fx, self.fy = np.meshgrid(self.Fc, self.Fr)
        self.x *= self.dx
        self.y *= self.dx
        self.fx /= self.dx
        self.fy /= self.dx

    def low_pass_filter(self, d2_current):
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

        # Apply LPF if needed
        self.LPF = self.low_pass_filter(d2_val) # Recalculate LPF for current d2_val
        if self.LPF is not None and np.sum(self.LPF) != 0:
            G = ne.evaluate("G * LPF", local_dict={'G':G, 'LPF':self.LPF})

        return L, G

    def angular_spectrum_method(self, initial_field, G):
        initial_field = pyfftw.interfaces.numpy_fft.ifftshift(initial_field.astype('complex64'))
        self.input_array[:] = initial_field
        self.fft_object()
        self.output_array[:] = pyfftw.interfaces.numpy_fft.fftshift(self.output_array)
        self.output_array[:] = self.output_array * G
        self.output_array[:] = pyfftw.interfaces.numpy_fft.ifftshift(self.output_array)
        self.ifft_object()
        ftr = pyfftw.interfaces.numpy_fft.fftshift(self.input_array)
        return ftr

def convert_np_array_to_base64_image(image_array):
    """Converts a NumPy array (grayscale) to a base64 encoded PNG image."""
    if image_array is None or image_array.size == 0:
        return ""

    min_val = np.min(image_array)
    max_val = np.max(image_array)

    if max_val - min_val > 1e-9:
        display_image = (image_array - min_val) / (max_val - min_val) * 255
    else:
        display_image = np.zeros_like(image_array)

    display_image = display_image.astype(np.uint8)

    img_pil = Image.fromarray(display_image)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route('/')
def index():
    initial_message = "Upload a hologram image and click 'Reconstruct' to begin."
    return render_template('index.html', initial_message=initial_message)

@app.route('/upload', methods=['POST'])
def upload_hologram():
    if 'hologram_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['hologram_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Generate a unique filename for the hologram data
        unique_filename = str(uuid.uuid4()) + '.npy'
        data_filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

        try:
            # Save the original image temporarily to load with plt.imread
            temp_img_filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(temp_img_filepath)

            hologram_loaded = plt.imread(temp_img_filepath)
            if hologram_loaded.ndim == 3:
                hologram_loaded = np.mean(hologram_loaded, axis=2) # Convert to grayscale
            hologram_data_np = hologram_loaded.astype(float)

            if hologram_data_np.size == 0:
                raise ValueError("Loaded hologram is empty.")

            # Save the processed NumPy array to disk
            np.save(data_filepath, hologram_data_np)

            # Store only the unique filename and shape in the session
            session['hologram_filepath'] = data_filepath
            session['hologram_shape'] = hologram_data_np.shape

            # Preview the uploaded hologram
            preview_image_b64 = convert_np_array_to_base64_image(hologram_loaded)
            nr0, nc0 = hologram_data_np.shape

            return jsonify({
                'success': True,
                'message': 'Hologram loaded successfully!',
                'preview_image': preview_image_b64,
                'dimension': f"{nr0}x{nc0}"
            })
        except Exception as e:
            return jsonify({'error': f"Could not load hologram: {e}"}), 500
        finally:
            # Clean up the original uploaded image file (the .npy version persists for reconstruction)
            if os.path.exists(temp_img_filepath):
                os.remove(temp_img_filepath)

@app.route('/reconstruct', methods=['POST'])
def reconstruct_hologram_web():
    # Retrieve only the filepath from the session
    hologram_filepath = session.get('hologram_filepath')
    hologram_shape = session.get('hologram_shape')

    if not hologram_filepath or not hologram_shape or not os.path.exists(hologram_filepath):
        return jsonify({'error': 'No hologram loaded or file not found. Please upload one first.'}), 400

    try:
        d2 = float(request.form.get('distance'))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid reconstruction distance provided.'}), 400

    start_total = time.perf_counter()

    try:
        # Load the numpy array from the temporary file
        hologram_np = np.load(hologram_filepath)
        # Verify shape just in case (though should be consistent)
        if hologram_np.shape != tuple(hologram_shape):
             return jsonify({'error': 'Hologram data corrupted or mismatch in shape.'}), 500

        reconstruction_core = HologramReconstructionCore(hologram_np)
        L, G = reconstruction_core.calculate_asm_parameters(d2)

        start_time_reconstruction = time.perf_counter()
        # Use the initial hologram from the core object which has been prepared
        ftr = reconstruction_core.angular_spectrum_method(reconstruction_core.hologram, G)
        end_time_reconstruction = time.perf_counter()

        if ftr is None:
            return jsonify({'error': 'Hologram reconstruction failed.'}), 500

        intensity = np.abs(ftr)**2
        reconstructed_image_b64 = convert_np_array_to_base64_image(intensity)

        end_total = time.perf_counter()

        return jsonify({
            'success': True,
            'reconstructed_image': reconstructed_image_b64,
            'total_time': f"{end_total - start_total:.4f} s",
            'propagation_time': f"{end_time_reconstruction - start_time_reconstruction:.4f} s"
        })

    except Exception as e:
        return jsonify({'error': f"An error occurred during reconstruction: {e}"}), 500
    # No finally block here to delete .npy, as it's needed for subsequent reconstructions
    # You might want a cleanup mechanism (e.g., a scheduled task) to remove old .npy files

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, threaded=False)