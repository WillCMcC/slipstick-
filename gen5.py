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
from scipy.signal import butter, lfilter, iirfilter

sampling_rate = 44100  # Sampling rate in Hz

current_sound = np.zeros(sampling_rate, dtype=np.float32)
current_freq = None  # Create a variable to store the current frequency
previous_params = None


def create_slider(window, min_value, max_value, resolution, orient, label):
    slider = tk.Scale(window, from_=min_value, to=max_value,
                      resolution=resolution, orient=orient, label=label)
    return slider

# Additional functions for low pass filter and resonance


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def distortion(signal, amount):
    return np.tanh(signal * amount) / np.tanh(amount)


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
        low_pass_filter_slider.get(),
        resonance_slider.get(),
        high_pass_filter_slider.get(),
        distortion_slider.get(),
    )
    if previous_params is None or current_params != previous_params:
        previous_params = current_params
        return True
    return False


def friction_force(velocity, friction_coeff):
    if velocity > 0:
        return -friction_coeff
    elif velocity < 0:
        return friction_coeff
    else:
        return 0


def acceleration(position, velocity, spring_const, damping_coeff, friction_coeff, mass):
    spring_force = -spring_const * position
    damping_force = -damping_coeff * velocity
    friction = friction_force(velocity, friction_coeff)

    net_force = spring_force + damping_force + friction
    return net_force / mass


def slip_stick_simulation(t, init_position, init_velocity, mass, spring_const, damping_coeff, friction_coeff):
    dt = t[1] - t[0]
    position = np.zeros_like(t)
    velocity = np.zeros_like(t)

    position[0] = init_position
    velocity[0] = init_velocity

    for i in range(1, len(t)):
        a = acceleration(position[i-1], velocity[i-1],
                         spring_const, damping_coeff, friction_coeff, mass)
        velocity[i] = velocity[i-1] + a * dt
        position[i] = position[i-1] + velocity[i-1] * dt
    return position, velocity


def generate_waveform(freq_radians, mod_radians, modulation_index, dt, t):
    if current_waveform_type.get() == "sine":
        carrier = np.sin(freq_radians * np.cumsum(dt * np.ones_like(t)))
    elif current_waveform_type.get() == "saw":
        carrier = np.cumsum(dt * freq_radians / np.pi) % 2 - 1
    elif current_waveform_type.get() == "square":
        carrier = np.sign(
            np.sin(freq_radians * np.cumsum(dt * np.ones_like(t))))
    modulator = modulation_index * \
        np.sin(mod_radians * np.cumsum(dt * np.ones_like(t)))
    waveform = (1 + modulator) * carrier
    return waveform


def generate_sound():
    duration = duration_slider.get()
    mass = mass_slider.get()
    stiffness = stiffness_slider.get()
    damping = damping_slider.get()
    friction = friction_slider.get()
    attack = attack_slider.get()
    decay = decay_slider.get()
    sustain = sustain_slider.get()
    release = release_slider.get()
    pitch = pitch_slider.get()
    modulation_index = modulation_index_slider.get()
    modulation_frequency = modulation_frequency_slider.get()

    slip_stick_params = (
        mass, stiffness, damping, friction, 0, 0)

    dt = 1.0 / sampling_rate
    t = np.arange(0, duration, dt)

    # Generate slip-stick signal
    position, velocity = slip_stick_simulation(
        t, 0, 1, mass, stiffness, damping, friction)
    slip_stick_signal = position

    # Create and apply an ADSR envelope
    total_samples = len(t)
    attack_samples = int(attack * total_samples)
    decay_samples = int(decay * total_samples)
    sustain_samples = max(0, total_samples - attack_samples -
                          decay_samples - int(release * total_samples))
    release_samples = total_samples - attack_samples - decay_samples - sustain_samples

    envelope = np.concatenate((
        np.linspace(0, 1, attack_samples),
        np.linspace(1, sustain, decay_samples),
        np.ones(sustain_samples) * sustain,
        np.linspace(sustain, 0, release_samples),
    ))

    slip_stick_signal = slip_stick_signal[:total_samples] * envelope

    # Generate the waveform
    freq_radians = pitch * 2 * np.pi
    mod_radians = modulation_frequency * 2 * np.pi
    waveform = generate_waveform(
        freq_radians, mod_radians, modulation_index, dt, t)

    # Combine the slip-stick signal with the waveform
    sound = waveform * slip_stick_signal

    # Add a Karplus-Strong algorithm to make it sound more punchy
    ks_buffer_size = int(sampling_rate // pitch)
    ks_buffer = np.random.uniform(-1, 1, size=ks_buffer_size)
    ks_output = np.zeros_like(sound)
    for i in range(len(sound)):
        ks_output[i] = ks_buffer[i % ks_buffer_size]
        ks_buffer[i % ks_buffer_size] = 0.5 * \
            (ks_output[i] + ks_buffer[(i - 1) % ks_buffer_size])
    sound = ks_output * slip_stick_signal

    # Apply the low pass filter
    cutoff_frequency = low_pass_filter_slider.get()
    sound = butter_lowpass_filter(sound, cutoff_frequency, sampling_rate)

    # Apply the high-pass filter
    cutoff_frequency_high = high_pass_filter_slider.get()
    sound = butter_highpass_filter(sound, cutoff_frequency_high, sampling_rate)

    # Apply the distortion effect
    distortion_amount = distortion_slider.get()
    sound = distortion(sound, distortion_amount)

    # Apply the resonance
    resonance = resonance_slider.get()
    sound = sound * resonance

    # Combine the Karplus-Strong output with the synthesized sound

    return sound


def normalize_signal(signal):
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal
    else:
        return signal / max_val


def generate_new_sound():
    global current_sound
    current_sound = normalize_signal(generate_sound())
    parameters_changed()
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


def stop_sound():
    sd.stop()


def save_sound():
    global current_sound

    file_name = f"slipstick_ui_output_{str(uuid.uuid4())}.wav"
    current_sound_normalized = normalize_signal(current_sound)

    write(file_name, sampling_rate,
          (current_sound_normalized * 32767).astype(np.int16))


def randomize_parameters():
    global current_sound
    duration_slider.set(np.random.uniform(.01, 2.0))
    mass_slider.set(np.random.uniform(0.001, 0.05))
    stiffness_slider.set(np.random.uniform(500.0, 5000.0))
    damping_slider.set(np.random.uniform(0.001, 0.1))
    pitch_slider.set(np.random.uniform(20.0, 200.0))
    attack_slider.set(np.random.uniform(0.005, 0.1))
    decay_slider.set(np.random.uniform(0.1, 0.5))
    sustain_slider.set(np.random.uniform(0.1, 0.8))
    release_slider.set(np.random.uniform(0.1, 1.5))
    modulation_index_slider.set(np.random.uniform(0, 10))
    modulation_frequency_slider.set(np.random.uniform(0, 1000))
    friction_slider.set(np.random.uniform(0.001, 0.01))
    distortion_slider.set(np.random.uniform(1, 50))

    # Randomize waveform type
    waveform_types = ["sine", "saw", "square"]
    current_waveform_type.set(np.random.choice(waveform_types))

    generate_new_sound()


def plot_sound():
    global current_sound

    fig.clear()
    ax = fig.add_subplot(111)

    _, _, _, im = ax.specgram(
        current_sound, NFFT=1024, Fs=sampling_rate, cmap="viridis", mode="magnitude")
    fig.colorbar(im, ax=ax)
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
        pitch_slider.set(frequency)
        play_sound(frequency)


window = tk.Tk()
window.title("Slipstick Synthesis")
plt.style.use("dark_background")

fig = plt.Figure(figsize=(10, 6), dpi=100)

canvas = FigureCanvasTkAgg(fig, master=window)
canvas.get_tk_widget().grid(row=0, rowspan=12, column=1)

duration_slider = create_slider(
    window, .01, 2, 0.01, tk.HORIZONTAL, "Duration (s)")
mass_slider = create_slider(window, 0.001, 0.05, 0.001, tk.HORIZONTAL, "Mass")
stiffness_slider = create_slider(
    window, 500, 5000, 10, tk.HORIZONTAL, "Spring Stiffness")
damping_slider = create_slider(
    window, 0.001, 0.1, 0.001, tk.HORIZONTAL, "Damping")
pitch_slider = create_slider(window, 20, 200, 1, tk.HORIZONTAL, "Pitch")
attack_slider = create_slider(
    window, 0.005, 0.1, 0.005, tk.HORIZONTAL, "Attack Time (s)")
decay_slider = create_slider(
    window, 0.1, 0.5, 0.1, tk.HORIZONTAL, "Decay Time (s)")
sustain_slider = create_slider(
    window, 0.1, 0.8, 0.1, tk.HORIZONTAL, "Sustain Level")
release_slider = create_slider(
    window, 0.1, 1.5, 0.1, tk.HORIZONTAL, "Release Time (s)")
modulation_index_slider = create_slider(
    window, 0.1, 10, 0.1, tk.HORIZONTAL, "Mod index")
modulation_frequency_slider = create_slider(
    window, 1, 1000, 10, tk.HORIZONTAL, "Mod freq")
friction_slider = create_slider(
    window, 0.001, 0.01, 0.001, tk.HORIZONTAL, "Friction")
low_pass_filter_slider = create_slider(
    window, 20, 20000, 10, tk.HORIZONTAL, "Low Pass Filter (Hz)")
resonance_slider = create_slider(
    window, 0.1, 10, 0.1, tk.HORIZONTAL, "Resonance")
high_pass_filter_slider = create_slider(
    window, 20, 20000, 10, tk.HORIZONTAL, "High Pass Filter (Hz)")
distortion_slider = create_slider(
    window, 1, 50, 1, tk.HORIZONTAL, "Distortion")

current_waveform_type = tk.StringVar(value="sine")

sine_wave_button = ttk.Radiobutton(
    window, text="Sine wave", variable=current_waveform_type, value="sine")
saw_wave_button = ttk.Radiobutton(
    window, text="Sawtooth wave", variable=current_waveform_type, value="saw")
square_wave_button = ttk.Radiobutton(
    window, text="Square wave", variable=current_waveform_type, value="square")

duration_slider.set(.1)
mass_slider.set(0.01)
stiffness_slider.set(1500)
damping_slider.set(0.08)
pitch_slider.set(100.0)
attack_slider.set(0.01)
decay_slider.set(0.2)
sustain_slider.set(0.1)
release_slider.set(0.8)
modulation_index_slider.set(1.0)
modulation_frequency_slider.set(50.0)
friction_slider.set(0.005)
low_pass_filter_slider.set(20000)  # Set the default value to 20000 Hz
resonance_slider.set(1)  # Set the default value to 1
high_pass_filter_slider.set(20)
distortion_slider.set(10)

play_button = ttk.Button(window, text="Play Sound", command=play_sound)
stop_button = ttk.Button(window, text="Stop Sound", command=stop_sound)
save_button = ttk.Button(window, text="Save Sound", command=save_sound)

randomize_button = ttk.Button(
    window, text="Randomize Parameters", command=randomize_parameters)

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
stop_button.grid(row=13, column=0)
save_button.grid(row=14, column=0)
randomize_button.grid(row=15, column=0)
sine_wave_button.grid(row=16, column=0)
saw_wave_button.grid(row=17, column=0)
square_wave_button.grid(row=18, column=0)
low_pass_filter_slider.grid(row=12, column=1)
resonance_slider.grid(row=13, column=1)
high_pass_filter_slider.grid(row=14, column=1)
distortion_slider.grid(row=15, column=1)

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

window.bind("<KeyPress>", keyboard_event)

window.mainloop()
