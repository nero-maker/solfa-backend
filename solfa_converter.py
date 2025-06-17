def convert_to_solfa(filepath):
    import os
    import warnings
    warnings.filterwarnings("ignore")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import contextlib
    from pydub import AudioSegment
    import librosa
    import numpy as np
    import scipy.signal
    import scipy.ndimage
    import crepe
    import pretty_midi
    from music21 import converter, pitch, key
    import subprocess
    import logging

    logging.basicConfig(level=logging.WARNING, format='[%(levelname)s] %(message)s')

    AudioSegment.converter = "C:/ffmpeg/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"
    AudioSegment.ffprobe = "C:/ffmpeg/ffmpeg-7.1.1-essentials_build/bin/ffprobe.exe"

    def separate_vocals(input_path, output_dir='separated_audio'):
        try:
            subprocess.run(
                ["C:/Users/Hp User/Desktop/py/.venv/Scripts/demucs.exe", input_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            filename = os.path.splitext(os.path.basename(input_path))[0]
            vocals_path = os.path.join("separated", "htdemucs", filename, "vocals.wav")
            if os.path.exists(vocals_path):
                return vocals_path
            else:
                raise FileNotFoundError("vocals.wav not found after Demucs separation.")
        except Exception as e:
            raise RuntimeError(f"Vocal separation failed: {e}")

    try:
        input_path = separate_vocals(filepath)

        y, sr = librosa.load(input_path, sr=16000)

        def bandpass_filter(y, sr, low=100, high=800):
            sos = scipy.signal.butter(10, [low, high], btype='band', fs=sr, output='sos')
            return scipy.signal.sosfilt(sos, y)

        y = bandpass_filter(y, sr)
        intervals = librosa.effects.split(y, top_db=20)

        def get_solfa_mapping(root):
            scale = [pitch.Pitch(root).transpose(i).name for i in [0, 2, 4, 5, 7, 9, 11]]
            solfa = ['do', 're', 'mi', 'fa', 'so', 'la', 'ti']
            return dict(zip(scale, solfa))

        @contextlib.contextmanager
        def suppress_output():
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    yield

        midi_notes_all = []
        for start, end in intervals:
            segment = y[start:end]
            if len(segment) < 2048:
                segment = np.pad(segment, (0, 2048 - len(segment)))
            audio_input = segment.astype(np.float32).reshape(-1, 1)
            with suppress_output():
                _, frequency, confidence, _ = crepe.predict(audio_input, sr=16000, step_size=20, viterbi=True)
            for f, c in zip(frequency, confidence):
                if c > 0.5 and f > 0:
                    midi_notes_all.append(int(librosa.hz_to_midi(f)))

        if not midi_notes_all:
            return []

        smoothed_all = scipy.ndimage.median_filter(midi_notes_all, size=5)

        detected_key = key.Key('G')
        solfa_mapping = get_solfa_mapping(detected_key.tonic.name)
        reference_midi = pitch.Pitch(detected_key.tonic.name).midi

        solfa_lines = []
        for start, end in intervals:
            segment = y[start:end]
            if len(segment) < 2048:
                segment = np.pad(segment, (0, 2048 - len(segment)))
            audio_input = segment.astype(np.float32).reshape(-1, 1)
            with suppress_output():
                _, frequency, confidence, _ = crepe.predict(audio_input, sr=16000, step_size=20, viterbi=True)
            raw_midi = [int(librosa.hz_to_midi(f)) for f, c in zip(frequency, confidence) if c > 0.3 and f > 0]
            if not raw_midi:
                continue
            smoothed_midi = scipy.ndimage.median_filter(raw_midi, size=5)
            notes = []
            for midi in smoothed_midi:
                transposed = midi - reference_midi + pitch.Pitch('C').midi
                name = librosa.midi_to_note(transposed)
                base = name[:-1] if name[-1].isdigit() else name
                base = base.replace('#', '').replace('b', '')
                if base in solfa_mapping:
                    notes.append(solfa_mapping[base])
            cleaned = []
            last = None
            for n in notes:
                if n != last:
                    cleaned.append(n)
                    last = n
            if cleaned:
                solfa_lines.append(cleaned)

        try:
            with suppress_output():
                midi_stream = pretty_midi.PrettyMIDI()
                instrument = pretty_midi.Instrument(program=0)
                time = 0.0
                duration = 0.5
                for midi_note in smoothed_all:
                    note = pretty_midi.Note(velocity=100, pitch=int(midi_note), start=time, end=time + duration)
                    instrument.notes.append(note)
                    time += duration
                midi_stream.instruments.append(instrument)
                midi_stream.write('temp_output.mid')
                score = converter.parse('temp_output.mid')
                score.write('lily.png', fp='sheet_music.png')
        except Exception:
            pass

        formatted_lines = []
        for line in solfa_lines:
            chunks = [' '.join([n.capitalize() for n in line[i:i + 6]]) for i in range(0, len(line), 6)]
            formatted_lines.extend(chunks)

        return formatted_lines

    finally:
        # === Clean up temporary files ===
        try:
            if os.path.exists(filepath):
                os.remove(filepath)

            filename = os.path.splitext(os.path.basename(filepath))[0]
            vocals_path = os.path.join("separated", "htdemucs", filename, "vocals.wav")
            demucs_dir = os.path.join("separated", "htdemucs", filename)
            if os.path.exists(vocals_path):
                os.remove(vocals_path)
            if os.path.isdir(demucs_dir):
                import shutil
                shutil.rmtree(demucs_dir)

            if os.path.exists('temp_output.mid'):
                os.remove('temp_output.mid')
            if os.path.exists('sheet_music.png'):
                os.remove('sheet_music.png')
        except Exception:
            pass  # Ignore any cleanup errors
