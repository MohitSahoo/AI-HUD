import numpy as np
from scipy.io import wavfile

def create_alert_sound():
    # Generate a simple alert sound (beep)
    sample_rate = 44100  # samples per second
    duration = 1.0  # seconds
    frequency = 440.0  # Hz (A4 note)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate sine wave
    note = np.sin(frequency * t * 2 * np.pi)
    
    # Add some harmonics for a more interesting sound
    note += 0.5 * np.sin(2 * frequency * t * 2 * np.pi)
    note += 0.25 * np.sin(3 * frequency * t * 2 * np.pi)
    
    # Normalize to 16-bit integer range
    note = np.int16(note * 32767)
    
    # Save as WAV file
    wavfile.write('alert.wav', sample_rate, note)

if __name__ == "__main__":
    create_alert_sound() 