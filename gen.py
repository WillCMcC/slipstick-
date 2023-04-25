import sounddevice as sd
from scipy.io.wavfile import write
from tkinter import ttk
import tkinter as tk
import numpy as np

sampling_rate = 44100  # Sampling rate in Hz

# Function to generate the sound based on the current parameters


def generate_sound():
    # Read the parameter values from the UI
    duration = float(duration_entry.get())
    mass = float(mass_entry.get())
    spring_stiffness = float(stiffness_entry.get())
    friction = float(friction_entry.get())
    attack_time = float(attack_entry.get())
    decay_time = float(decay_entry.get())
    sustain_level = float(sustain_entry.get())
    release_time = float(release_entry.get())

    # Time step for numerical integration
    dt = 1.0 / sampling_rate

    # Initialize position, velocity, and acceleration
    position = 1.0
    velocity = 0.0
    acceleration = 0.0

    # Prepare the output array
    output = np.zeros(int(sampling_rate * duration))

    # Perform the slipstick algorithm using the Euler method
    for i in range(len(output)):
        # Calculate the spring force and friction force
        spring_force = -spring_stiffness * position
        friction_force = -friction * velocity if velocity != 0 else 0

        # Update acceleration based on the total force
        acceleration = (spring_force + friction_force) / mass

        # Update position and velocity using the Euler method
        position += velocity * dt
        velocity += acceleration * dt

        # Store the position value as the output sound
        output[i] = position

    # Apply an ADSR envelope
    attack_samples = int(attack_time * sampling_rate)
    decay_samples = int(decay_time * sampling_rate)
    release_samples = int(release_time * sampling_rate)
    sustain_samples = len(output) - attack_samples - \
        decay_samples - release_samples

    envelope = np.concatenate([
        np.linspace(0, 1, attack_samples),
        np.linspace(1, sustain_level, decay_samples),
        np.ones(sustain_samples) * sustain_level,
        np.linspace(sustain_level, 0, release_samples),
    ])

    output = output * envelope

    # Normalize the output to the range [-1, 1]
    output = output / np.max(np.abs(output))

    return output

# Function to play the sound


def play_sound():
    output = generate_sound()
    sd.play(output, samplerate=sampling_rate)

# Function to save the sound


def save_sound():
    output = generate_sound()
    write("slipstick_ui_output.wav", sampling_rate,
          (output * 32767).astype(np.int16))


# Create the main window
window = tk.Tk()
window.title("Slipstick Synthesis")

# Create labels and entry fields for the parameters
duration_label = ttk.Label(window, text="Duration (s):")
duration_entry = ttk.Entry(window, width=10)
duration_entry.insert(0, "4.0")

mass_label = ttk.Label(window, text="Mass:")
mass_entry = ttk.Entry(window, width=10)
mass_entry.insert(0, "0.01")

stiffness_label = ttk.Label(window, text="Spring Stiffness:")
stiffness_entry = ttk.Entry(window, width=10)
stiffness_entry.insert(0, "250.0")

friction_label = ttk.Label(window, text="Friction:")
friction_entry = ttk.Entry(window, width=10)
friction_entry.insert(0, "0.05")

attack_label = ttk.Label(window, text="Attack Time (s):")
attack_entry = ttk.Entry(window, width=10)
attack_entry.insert(0, "0.01")

decay_label = ttk.Label(window, text="Decay Time (s):")
decay_entry = ttk.Entry(window, width=10)
decay_entry.insert(0, "0.2")

sustain_label = ttk.Label(window, text="Sustain Level:")
sustain_entry = ttk.Entry(window, width=10)
sustain_entry.insert(0, "0.4")

release_label = ttk.Label(window, text="Release Time (s):")
release_entry = ttk.Entry(window, width=10)
release_entry.insert(0, "0.8")

# Create the "Play Sound" button
play_button = ttk.Button(window, text="Play Sound", command=play_sound)

# Create the "Save Sound" button
save_button = ttk.Button(window, text="Save Sound", command=save_sound)

# Place the UI elements on a grid
duration_label.grid(row=0, column=0, sticky="e")
duration_entry.grid(row=0, column=1)

mass_label.grid(row=1, column=0, sticky="e")
mass_entry.grid(row=1, column=1)

stiffness_label.grid(row=2, column=0, sticky="e")
stiffness_entry.grid(row=2, column=1)

friction_label.grid(row=3, column=0, sticky="e")
friction_entry.grid(row=3, column=1)

attack_label.grid(row=4, column=0, sticky="e")
attack_entry.grid(row=4, column=1)

decay_label.grid(row=5, column=0, sticky="e")
decay_entry.grid(row=5, column=1)

sustain_label.grid(row=6, column=0, sticky="e")
sustain_entry.grid(row=6, column=1)

release_label.grid(row=7, column=0, sticky="e")
release_entry.grid(row=7, column=1)

play_button.grid(row=8, column=0, columnspan=2)
save_button.grid(row=9, column=0, columnspan=2)

# Run the main loop
window.mainloop()
