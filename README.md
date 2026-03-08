# Automatic Piano Transcription via Onsets and Frames

This repository contains a technical implementation of an Automatic Piano Transcription (APT) system. The framework is designed to convert raw polyphonic piano audio into MIDI representations by jointly modeling note onsets, offsets, sustained frames, and velocities using a multi-head deep neural network.



## Research Objective

The project explores the efficacy of temporal modeling and feature fusion in the context of piano acoustics. By utilizing the "Onsets and Frames" paradigm, the system aims to provide higher temporal resolution and note consistency compared to standard frame-level classification models.

### Architectural Components
* **Onset Detection**: A dedicated head for identifying the attack phase of a note.
* **Offset Detection**: A head for identifying the release phase to improve duration accuracy.
* **Frame Activation**: Sustain-state modeling to maintain note continuity.
* **Velocity Regression**: Estimating the dynamic intensity of each keystroke.
* **Recurrent Fusion**: A Bi-directional LSTM layer that integrates logits from multiple heads for final transcription.

---

## Technical Specifications

The system is calibrated to the MAESTRO dataset standards to ensure academic reproducibility.

| Parameter | Value |
| :--- | :--- |
| Sampling Rate | 16,000 Hz |
| STFT Window Size | 2,048 samples |
| Hop Length | 512 samples |
| Input Features | 229 Log-Mel frequency bins |
| Output Classes | 88 MIDI notes (A0 to C8) |
| Recurrent Layer | Bi-directional LSTM |



---

## Project Structure

* `src/`: Modular implementation of the framework.
    * `model.py`: Neural network architecture and fusion logic.
    * `modules.py`: Custom layers including Acoustic Extractors and BiLSTMs.
    * `dataset.py`: Data loading and preprocessing for the MAESTRO dataset.
    * `mel.py`: Signal processing pipeline for spectral feature extraction.
    * `midi.py`: MIDI encoding/decoding and ground-truth generation.
* `train.py`: Training pipeline supporting Automatic Mixed Precision (AMP).
* `inference.py`: Transcription utility for transforming .wav files to .mid.

---

## Current Development Status

This project is currently in the active development and architectural validation phase.

* **Model Implementation**: Core architecture and tensor dimensions are defined and validated.
* **Inference Pipeline**: The end-to-end chain from raw audio to MIDI file generation is established.
* **Training Readiness**: The data loading and loss functions are implemented; initial training runs on the MAESTRO dataset are pending.

