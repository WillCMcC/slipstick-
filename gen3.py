import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.io.wavfile import write
import sounddevice as sd
import matplotlib.pyplot as plt
import numba
from scipy.integrate import odeint
import keyboard
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import uuid


sampling_rate = 44100  # Sampling rate in Hz

current_sound = None  # Create a variable to store the current sound
current_freq = None  # Create a variable to store the current frequency
previous_params = None

# Function to generate the sound based on the current parameters


def parameters_changed():
    global previous_params
    current_params = (
        duration_slider.get(), feedback_slider.get(), freq_slider.get(), noise_bandwidth_slider.get(), noise_extent_slider.get(), attack_slider.get(), decay_slider.get(
        ), sustain_slider.get(), release_slider.get(), distortion_slider.get()
    )
    if previous_params is None or current_params != previous_params:
        previous_params = current_params
        return True
    return False


def generate_sound(duration, feedback, freq, noise_bandwidth, noise_extent, attack_time, decay_time, sustain_level, release_time, sampling_rate, distortion_amount):
    dt = 1.0 / sampling_rate
    t = np.arange(0, duration, dt)

    # Generate noise signal
    noise_signal = np.random.uniform(-noise_extent, noise_extent, size=len(t))
    noise_signal = np.convolve(
        noise_signal, np.ones(noise_bandwidth), mode='same')

    # Karplus-Strong Algorithm
    buffer_size = int(sampling_rate / freq)
    buffer = np.zeros(buffer_size)
    buffer[:buffer_size] = noise_signal[:buffer_size]
    output = np.zeros(len(t))

    for i in range(buffer_size, len(t)):
        buffer_sample = feedback * 0.5 * \
            (buffer[i % buffer_size] + buffer[(i - 1) % buffer_size])
        buffer[i % buffer_size] = buffer_sample
        output[i] = buffer_sample

    # Apply an ADSR envelope
    total_samples = int(duration * sampling_rate)
    attack_samples = int(attack_time * total_samples)
    decay_samples = int(decay_time * total_samples)
    sustain_samples = max(0, total_samples - attack_samples -
                          decay_samples - int(release_time * total_samples))
    release_samples = total_samples - attack_samples - decay_samples - sustain_samples

    envelope = np.concatenate((
        np.linspace(0, 1, attack_samples),
        np.linspace(1, sustain_level, decay_samples),
        np.ones(sustain_samples) * sustain_level,
        np.linspace(sustain_level, 0, release_samples),
    ))

    output = output[:total_samples] * envelope

    output = distortion(output, distortion_amount)

    return output
# Function to play the sound


def generate_new_sound():
    global current_sound
    current_sound = generate_sound(duration_slider.get(), feedback_slider.get(), freq_slider.get(), noise_bandwidth_slider.get(), noise_extent_slider.get(), attack_slider.get(), decay_slider.get(
    ), sustain_slider.get(), release_slider.get(), sampling_rate, distortion_slider.get())
    plot_sound()


def play_sound(freq=None):
    global current_sound
    global current_freq
    if freq is not None:
        freq_slider.set(freq)
    # Regenerate the sound with updated frequency value
    if freq is not current_freq or parameters_changed():
        generate_new_sound()
    current_freq = freq
    sd.play(current_sound, samplerate=sampling_rate)
    plot_sound()


# Function to save the sound


def save_sound():
    global current_sound
    file_name = f"slipstick_ui_output_{str(uuid.uuid4())}.wav"
    current_sound_normalized = current_sound / np.max(np.abs(current_sound))

    write(file_name, sampling_rate,
          (current_sound_normalized * 32767).astype(np.int16))

# Function to randomize the parameter values


def randomize_parameters():
    duration_slider.set(np.random.uniform(0.02, 2.0))
    feedback_slider.set(np.random.uniform(0.9, 0.999))
    freq_slider.set(np.random.uniform(0, 200))
    noise_bandwidth_slider.set(np.random.uniform(50, 500))
    noise_extent_slider.set(np.random.uniform(0.3, 0.8))
    attack_slider.set(np.random.uniform(0.005, 0.02))
    decay_slider.set(np.random.uniform(0.1, 0.3))
    sustain_slider.set(np.random.uniform(0.2, 0.6))
    release_slider.set(np.random.uniform(0.1, 0.5))
    distortion_slider.set(np.random.uniform(2, 8))
    generate_new_sound()  # Add this line to generate a new sound after randomizing parameters


def distortion(sound, amount):
    return np.tanh(amount * sound)


def plot_sound():
    global current_sound
    fig.clear()
    ax = fig.add_subplot(111)

    data = np.maximum(current_sound, 1e-8)
    spectrogram = ax.specgram(
        data, NFFT=1024, Fs=sampling_rate, cmap="viridis", noverlap=900)
    ax.set_xlabel("Time (s)", color="white", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", color="white", fontsize=12)

    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    canvas.draw()


def keyboard_event(e):
    key = e.char.lower()
    if key == 'r':
        randomize_parameters()
    elif key in keyboard_mapping:
        frequency = keyboard_mapping[key]
        play_sound(frequency)


# Create the main window
window = tk.Tk()
window.title("Slipstick Synthesis")
plt.style.use("dark_background")

# Create the chart to visualize the sound
fig = plt.Figure(figsize=(10, 6), dpi=100)

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().grid(row=0, rowspan=12, column=1)

# Create sliders for the parameters
duration_slider = tk.Scale(
    window, from_=.01, to=10, resolution=0.1, orient=tk.HORIZONTAL, label="Duration (s)")


attack_slider = tk.Scale(window, from_=0.005, to=0.1, resolution=0.005,
                         orient=tk.HORIZONTAL, label="Attack Time (s)")
decay_slider = tk.Scale(window, from_=0.1, to=0.5, resolution=0.1,
                        orient=tk.HORIZONTAL, label="Decay Time (s)")
sustain_slider = tk.Scale(window, from_=0.1, to=0.8,
                          resolution=0.1, orient=tk.HORIZONTAL, label="Sustain Level")
release_slider = tk.Scale(window, from_=0.1, to=1.5, resolution=0.1,
                          orient=tk.HORIZONTAL, label="Release Time (s)")

friction_slider = tk.Scale(window, from_=0.001, to=0.1,
                           resolution=0.001, orient=tk.HORIZONTAL, label="Friction")
feedback_slider = tk.Scale(window, from_=0.5, to=0.999,
                           resolution=0.001, orient=tk.HORIZONTAL, label="Feedback")
freq_slider = tk.Scale(window, from_=0, to=200,
                       resolution=1, orient=tk.HORIZONTAL, label="Frequency")
noise_bandwidth_slider = tk.Scale(window, from_=1, to=1000,
                                  resolution=1, orient=tk.HORIZONTAL, label="Noise Bandwidth")
noise_extent_slider = tk.Scale(window, from_=0.001, to=1,
                               resolution=0.001, orient=tk.HORIZONTAL, label="Noise Extent")
distortion_slider = tk.Scale(window, from_=1, to=100,
                             resolution=1, orient=tk.HORIZONTAL, label="Distortion Amount")

# Set default slider values
duration_slider.set(1)
attack_slider.set(0.001)
decay_slider.set(0.15)
sustain_slider.set(0.2)
release_slider.set(0.8)
friction_slider.set(0.02)
feedback_slider.set(0.92)
freq_slider.set(300)
noise_bandwidth_slider.set(600)
noise_extent_slider.set(0.6)
distortion_slider.set(5)

# Create the "Play Sound" button and "Save Sound" button
play_button = ttk.Button(window, text="Play Sound", command=play_sound)
save_button = ttk.Button(window, text="Save Sound", command=save_sound)

# Create the "Randomize" button
randomize_button = ttk.Button(
    window, text="Randomize Parameters", command=randomize_parameters)

# Place the UI elements on a grid
duration_slider.grid(row=0, column=0)
feedback_slider.grid(row=1, column=0)
freq_slider.grid(row=2, column=0)
noise_bandwidth_slider.grid(row=3, column=0)
distortion_slider.grid(row=4, column=0)
noise_extent_slider.grid(row=5, column=0)
friction_slider.grid(row=6, column=0)
attack_slider.grid(row=7, column=0)
decay_slider.grid(row=8, column=0)
sustain_slider.grid(row=9, column=0)
release_slider.grid(row=10, column=0)
play_button.grid(row=11, column=0)
save_button.grid(row=12, column=0)
randomize_button.grid(row=13, column=0)

keyboard_mapping = {
    'a': 261.63,  # C4
    'w': 277.18,  # C#4
    's': 293.66,  # D4
    'e': 311.13,  # D#4
    'd': 329.63,  # E4
    'f': 349.23,  # F4
    't': 369.99,  # F#4
    'g': 392.00,  # G4
    'y': 415.30,  # G#4
    'h': 440.00,  # A4
    'u': 466.16,  # A#4
    'j': 493.88,  # B4
    'k': 523.25,  # C5
}

# Bind keyboard events to the main window
window.bind("<KeyPress>", keyboard_event)


# Call generate_new_sound at the end of the script to generate the initial sound
generate_new_sound()

# Run the main loop
window.mainloop()
