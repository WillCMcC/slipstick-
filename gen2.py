import numpy as np
import tkinter as tk
from tkinter import ttk
from scipy.io.wavfile import write
import sounddevice as sd
import matplotlib.pyplot as plt
import numba
from scipy.integrate import odeint
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import uuid

sampling_rate = 44100  # Sampling rate in Hz

current_sound = np.zeros(sampling_rate, dtype=np.float32)
current_freq = None  # Create a variable to store the current frequency
previous_params = None

# Function to generate the sound based on the current parameters


def parameters_changed():
    global previous_params
    current_params = (
        duration_slider.get(),
        mass_slider.get(),
        stiffness_slider.get(),
        damping_slider.get(),
        friction_slider.get(),
        attack_slider.get(),
        decay_slider.get(),
        sustain_slider.get(),
        release_slider.get(),
        pitch_slider.get(),
        modulation_index_slider.get(),
        modulation_frequency_slider.get(),
    )
    if previous_params is None or current_params != previous_params:
        previous_params = current_params
        return True
    return False


def vibrating_plate_equation(y, t, stiffness, damping, mass, friction):
    pos_x, pos_y, vel_x, vel_y = y
    acceleration_x = -stiffness * pos_x - damping * vel_x + \
        total_force(pos_x, vel_x, stiffness, friction, mass, 9.81)
    acceleration_y = -stiffness * pos_y - damping * vel_y + \
        total_force(pos_y, vel_y, stiffness, friction, mass, 9.81)
    return [vel_x, vel_y, acceleration_x, acceleration_y]


def total_force(pos, vel, spring_stiffness, friction, mass, gravitational_acceleration):
    spring_force = -spring_stiffness * pos
    friction_force = 0

    static_friction_force = friction * mass * gravitational_acceleration

    # If the spring force is greater than the static friction force, apply friction in the opposite direction of motion
    if np.abs(spring_force) > static_friction_force:
        friction_force = -np.sign(spring_force) * static_friction_force

    return spring_force + friction_force


def generate_sound(duration, mass, stiffness, damping, friction, attack_time, decay_time, sustain_level, release_time, pitch, modulation_index, modulation_frequency, sampling_rate):
    dt = 1.0 / sampling_rate
    t = np.arange(0, duration, dt)

    # Initial conditions: pos_x, pos_y, vel_x, vel_y
    init_conditions = [1.0, 1.0, 0.0, 0.0]
    sol = odeint(vibrating_plate_equation, init_conditions,
                 t, args=(stiffness, damping, mass, friction))
    position = sol[:, :2]

    # FM synthesis
    modulator = modulation_index * np.sin(2 * np.pi * modulation_frequency * t)
    output = np.sin(2 * np.pi * (pitch + modulator) * t)

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

    output = (output[:total_samples] * envelope).astype(np.float32)

    return output

# Function to play the sound


def generate_new_sound():
    global current_sound
    current_sound = generate_sound(duration_slider.get(), mass_slider.get(), stiffness_slider.get(), damping_slider.get(), friction_slider.get(), attack_slider.get(), decay_slider.get(
    ), sustain_slider.get(), release_slider.get(), pitch_slider.get(), modulation_index_slider.get(), modulation_frequency_slider.get(), sampling_rate)
    plot_sound()


def play_sound(freq=None):
    global current_sound
    global current_freq

    if freq is not None:
        pitch_slider.set(freq)
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
    global current_sound
    duration_slider.set(np.random.uniform(2.0, 6.0))
    mass_slider.set(np.random.uniform(0.001, 0.05))
    stiffness_slider.set(np.random.uniform(100.0, 500.0))
    damping_slider.set(np.random.uniform(0.01, 0.1))
    pitch_slider.set(np.random.uniform(20.0, 200.0))
    attack_slider.set(np.random.uniform(0.005, 0.1))
    decay_slider.set(np.random.uniform(0.1, 0.5))
    sustain_slider.set(np.random.uniform(0.1, 0.8))
    release_slider.set(np.random.uniform(0.1, 1.5))
    modulation_index_slider.set(np.random.uniform(0, 10))
    modulation_frequency_slider.set(np.random.uniform(0, 1000))
    friction_slider.set(np.random.uniform(0.001, 0.1))
    generate_new_sound()


def plot_sound():
    global current_sound

    fig.clear()
    ax = fig.add_subplot(111)

    # Create spectrogram
    spectrogram = ax.specgram(
        current_sound, NFFT=1024, Fs=sampling_rate, cmap="viridis")
    ax.set_xlabel("Time (s)", color="white", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", color="white", fontsize=12)

    # Set the tick labels to white
    ax.tick_params(axis="x", colors="white")
    ax.tick_params(axis="y", colors="white")

    canvas.draw()


def keyboard_event(e):
    key = e.char.lower()
    if key == 'r':
        randomize_parameters()
    elif key in keyboard_mapping:
        frequency = keyboard_mapping[key]
        pitch_slider.set(frequency)
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
    window, from_=1, to=10, resolution=0.1, orient=tk.HORIZONTAL, label="Duration (s)")
mass_slider = tk.Scale(window, from_=0.001, to=0.05,
                       resolution=0.001, orient=tk.HORIZONTAL, label="Mass")
stiffness_slider = tk.Scale(window, from_=100, to=500,
                            resolution=1, orient=tk.HORIZONTAL, label="Spring Stiffness")

damping_slider = tk.Scale(window, from_=0.01, to=0.1,
                          resolution=0.01, orient=tk.HORIZONTAL, label="Damping")

pitch_slider = tk.Scale(window, from_=20, to=200,
                        resolution=1, orient=tk.HORIZONTAL, label="Pitch")
attack_slider = tk.Scale(window, from_=0.005, to=0.1, resolution=0.005,
                         orient=tk.HORIZONTAL, label="Attack Time (s)")
decay_slider = tk.Scale(window, from_=0.1, to=0.5, resolution=0.1,
                        orient=tk.HORIZONTAL, label="Decay Time (s)")
sustain_slider = tk.Scale(window, from_=0.1, to=0.8,
                          resolution=0.1, orient=tk.HORIZONTAL, label="Sustain Level")
release_slider = tk.Scale(window, from_=0.1, to=1.5, resolution=0.1,
                          orient=tk.HORIZONTAL, label="Release Time (s)")
modulation_index_slider = tk.Scale(window, from_=0.1, to=10, resolution=0.1,
                                   orient=tk.HORIZONTAL, label="Mod index")
modulation_frequency_slider = tk.Scale(window, from_=1, to=1000, resolution=10,
                                       orient=tk.HORIZONTAL, label="Mod freq")
friction_slider = tk.Scale(window, from_=0.001, to=0.1,
                           resolution=0.001, orient=tk.HORIZONTAL, label="Friction")

# Set default slider values
duration_slider.set(4.0)
mass_slider.set(0.01)
stiffness_slider.set(250.0)
damping_slider.set(0.05)
pitch_slider.set(100.0)
attack_slider.set(0.01)
decay_slider.set(0.2)
sustain_slider.set(0.4)
release_slider.set(0.8)
modulation_index_slider.set(5.0)
modulation_frequency_slider.set(100.0)
friction_slider.set(0.01)

# Create the "Play Sound" button and "Save Sound" button
play_button = ttk.Button(window, text="Play Sound", command=play_sound)
save_button = ttk.Button(window, text="Save Sound", command=save_sound)

# Create the "Randomize" button
randomize_button = ttk.Button(
    window, text="Randomize Parameters", command=randomize_parameters)

# Place the UI elements on a grid
duration_slider.grid(row=0, column=0)
mass_slider.grid(row=1, column=0)
stiffness_slider.grid(row=2, column=0)
damping_slider.grid(row=3, column=0)
friction_slider.grid(row=4, column=0)
pitch_slider.grid(row=5, column=0)
attack_slider.grid(row=6, column=0)
decay_slider.grid(row=7, column=0)
sustain_slider.grid(row=8, column=0)
release_slider.grid(row=9, column=0)
modulation_index_slider.grid(row=10, column=0)
modulation_frequency_slider.grid(row=11, column=0)

play_button.grid(row=12, column=0)
save_button.grid(row=13, column=0)
randomize_button.grid(row=14, column=0)

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

# Run the main loop
window.mainloop()
