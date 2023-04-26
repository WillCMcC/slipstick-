# Slipstick Synthesis

This project is a Python-based synthesizer that allows you to generate sounds using slip-stick simulation and various sound waveform types such as sine, sawtooth, and square waves. The synthesizer also provides various sliders to control the sound parameters like duration, mass, stiffness, damping, friction, pitch, attack, decay, sustain, release, modulation index, modulation frequency, low-pass filter, resonance, high-pass filter, and distortion.

There are 4 synths, gen2, gen3, and gen4. gen2 is weird. gen3 is plucky. gen4 is fancy, gen5 is KP gen 4. Im not solid on any of the physics so this is a work in progress and any feedback or PRs are greatly appreciated! 

## Installation

To install and run this project, you need to have Python 3 installed on your computer. You also need to install the following Python libraries:

- numpy
- scipy
- tkinter
- sounddevice
- matplotlib
- numba

You can install these libraries using pip by running the following command:

```bash
pip install numpy scipy tkinter sounddevice matplotlib numba
```

## Usage

To run the synthesizer, you can simply run the desired Python script ( `gen2.py`, or `gen3.py`, `gen4.py`) using Python. For example:

```bash
python gen2.py
```

This will launch the synthesizer interface with various sliders to control the sound parameters. You can adjust these sliders and click the "Play Sound" button to listen to the generated sound. You can also use the "Stop Sound" button to stop the sound playback and the "Save Sound" button to save the generated sound as a WAV file.

The "Randomize Parameters" button will randomly adjust the sliders, allowing you to explore various sound combinations. Additionally, you can use your computer keyboard to play notes with the following key mappings:

- A: C4
- W: C#4
- S: D4
- E: D#4
- D: E4
- F: F4
- T: F#4
- G: G4
- Y: G#4
- H: A4
- U: A#4
- J: B4
- K: C5

Pressing the corresponding key will generate and play the sound with the specified frequency.

## License

This project is open-source and available for anyone to use, modify, or distribute. Please feel free to contribute and improve the project.