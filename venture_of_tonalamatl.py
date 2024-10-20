#!/usr/bin/env python3

from __future__ import annotations

import math

from fractions import Fraction
from numbers import Rational
from time import sleep

import numpy as np
import sounddevice as sd

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from numbers import Real
    from typing import Literal
    from numpy.typing import NDArray


SAMPLE_RATE = 48000

_NOTES = {
    'C-': 12,
    'C#': 13, 'Db': 13,
    'D-': 14,
    'D#': 15, 'Eb': 15,
    'E-': 16,
    'F-': 17,
    'F#': 18, 'Gb': 18,
    'G-': 19,
    'G#': 20, 'Ab': 20,
    'A-': 21,
    'A#': 22, 'Bb': 22,
    'B-': 23,
}


def note_to_midi_pitch(note: str) -> int | None:

    if len(note) != 3:
        return None

    name, octave = note[0].upper() + note[1].lower(), note[2]
    if name not in _NOTES or not octave.isdecimal():
        return None

    return _NOTES[name] + int(octave) * 12


def silence(sample_duration: Real) -> NDArray[np.float32]:

    return np.zeros(round(sample_duration * SAMPLE_RATE), dtype=np.float32)


def sinewave(sample_duration: Real, frequency: Real) -> NDArray[np.float32]:

    return np.sin(np.linspace(0.0, 2 * np.pi * sample_duration * frequency, num = round(sample_duration * SAMPLE_RATE), endpoint=False, dtype=np.float32))


def schedule(sample_duration: Real, sched: Sequence[tuple[Literal['ramp', 'lin', 'linear'], Real, Real]], start_level: Real = 0) -> NDArray[np.float32]:

    result = np.empty(round(sample_duration * SAMPLE_RATE), dtype=np.float32)

    t1, left, i1 = 0, 0, 0
    level = start_level
    for typ, t2, new_level in sched:
        if i1 >= len(result):
            break

        right = t2 * SAMPLE_RATE

        if t2 == t1 or not (isinstance(t1, Rational) and isinstance(t2, Rational)) and abs(t2 - t1) < 0.0005:
            level = new_level
            left = right
            continue

        assert t2 > t1

        i2 = min(math.ceil(right), len(result))
        if i2 <= i1:
            level = new_level
            left, t1 = right, t2
            continue

        match typ:
            case 'ramp' | 'lin' | 'linear':
                result[i1:i2] = np.linspace(
                    (i1 - left) / (right - left) * (new_level - level) + level,
                    (i2 - left) / (right - left) * (new_level - level) + level,
                    num=i2 - i1, endpoint=False, dtype=np.float32
                )
            case _:
                raise NotImplementedError(f"Unknown curve type {typ!r}")

        level = new_level
        left, t1, i1 = right, t2, i2

    if i1 < len(result):
        result[i1:] = new_level

    return result


def envelope(sample_duration: Real, on_duration: Real = 0.2, attack_duration: Real = 0.05, decay_duration: Real = 0.4, sustain_level: Real = 0.5, release_duration: Real = 0.4) -> NDArray[np.float32]:

    if on_duration <= 0:
        return silence(sample_duration)

    if on_duration <= attack_duration:
        return schedule(sample_duration, [
            ('linear', on_duration, last_level := on_duration / attack_duration),
            ('linear', on_duration + release_duration * (last_level / sustain_level), 0),
        ])

    sustain_t0 = attack_duration + decay_duration
    if on_duration <= sustain_t0:
        return schedule(sample_duration, [
            ('linear', attack_duration, 1),
            ('linear', on_duration, last_level := 1 - (1 - sustain_level) * (on_duration - attack_duration) / decay_duration),
            ('linear', on_duration + release_duration * (last_level / sustain_level), 0),
        ])

    return schedule(sample_duration, [
        ('linear', attack_duration, 1),
        ('linear', sustain_t0, sustain_level),
        ('linear', on_duration, sustain_level),
        ('linear', on_duration + release_duration, 0),
    ])


class AudioDeviceEvaluator:

    def __init__(self, target_channels=2, min_channels=None, max_channels=8, target_latency=0.008):

        self._target_channels = target_channels
        self._min_channels = min(min_channels, target_channels) if min_channels is not None else target_channels
        self._max_channels = max(max_channels, target_channels) if max_channels is not None else math.inf
        self._target_latency = target_latency

    def key_func(self, device) -> float:

        weight = 1.0
        idx = device['index']

        channels = device['max_output_channels']
        if channels < self._min_channels:
            return (-math.inf, channels, ~idx)
        ratio = min(channels, self._max_channels) / self._target_channels
        weight *= ratio * ratio if ratio <= 1 else 1 + math.log2(ratio) / 3

        latency = device['default_low_output_latency']
        if latency <= 0:
            return (0.0, channels, ~idx)
        ratio = self._target_latency / latency
        weight *= ratio if ratio <= 1 else 1 + math.log(ratio)

        hostapi = device['hostapi']
        if idx == sd.query_hostapis(hostapi)['default_output_device']:
            weight *= (2.5 + hostapi) / (1.5 + hostapi)

        return (weight, channels, ~idx)


def trim_trailing_zeros(sound: NDArray[np.float32]) -> None:

    chunk_size = 1
    start = len(sound)
    while True:
        start -= chunk_size
        if start >= 0 and np.array_equiv(sound[start:], 0):
            sound.resize((start, *sound.shape[1:]), refcheck=False)
            chunk_size *= 2
            continue

        while True:
            chunk_size //= 2
            if not chunk_size:
                return

            start += chunk_size
            if start >= 0 and np.array_equiv(sound[start:], 0):
                sound.resize((start, *sound.shape[1:]), refcheck=False)
                chunk_size //= 2
                if not chunk_size:
                    return

                break


def make_song(
        melody: Iterable[str], *,
        tempo: Real = 120, beat: Real = Fraction(1, 4), gap: Real = 0.02,
        tuning_frequency: float = 440, tuning_note: str = 'A-4',
        detune_cents: float = 0,
) -> NDArray[np.float32]:

    whole_note_duration = (
        Fraction(beat).limit_denominator() ** -1 *
        60 / Fraction(tempo).limit_denominator()
    )

    gap = Fraction(gap).limit_denominator()

    reference_pitch = note_to_midi_pitch(tuning_note)
    detune = detune_cents / 1200  # cents / 100 / 12, i.e. a fraction of an octave

    song = silence(15)

    t, end = 0, 0
    for item in melody:
        start = round(t * SAMPLE_RATE)
        if start >= len(song):
            break

        duration, note = item.split(' ')
        duration = Fraction(duration) * whole_note_duration
        t += duration

        pitch = note_to_midi_pitch(note)
        if pitch is None:
            continue

        frequency = tuning_frequency * 2 ** ((pitch - reference_pitch) / 12 + detune)
        sound = (0.1 * sinewave(duration + 5, frequency) + 0.06 * sinewave(duration + 5, frequency * 1.99)) * envelope(duration + 5, duration - gap)
        end = start + len(sound)

        if end > len(song):
            song.resize((max(end, len(song) * 2), *song.shape[1:]), refcheck=False)

        song[start : end] += sound

    if end < len(song):
        song.resize((end, *song.shape[1:]), refcheck=False)

    trim_trailing_zeros(song)
    return song


def main() -> None:

    song = make_song([
        # '11/16 ---',
        '1/16 G-5',
        '1/16 E-5',
        '1/16 G-5',
        '1/16 A-5',
        '1/16 A#5',

        '3/16 B-5',
        '1/16 G-5',
        '2/16 ---',
        '2/16 E-5',
        '2/16 ---',
        '1/16 E-5',
        '1/16 F#5',
        '2/16 E-5',
        '2/16 B-4',

        '3/16 D-5',
        '13/16 C-5',

        '3/16 A-5',
        '1/16 F#5',
        '2/16 ---',
        '2/16 D-5',
        '2/16 ---',
        '2/16 C-5',
        '2/16 B-4',
        '2/16 A#4',

        '6/16 B-4',
        '2/16 C#5',
        '1/16 D#5',
        '2/16 E-5',
        '1/16 F#5',
        '2/16 G-5',
        '2/16 A-5',

        '3/16 B-5',
        '1/16 G-5',
        '2/16 ---',
        '2/16 E-5',
        '2/16 ---',
        '1/16 E-5',
        '1/16 F#5',
        '2/16 E-5',
        '2/16 B-4',

        '2/16 D-5',
        '1/16 D-5',
        '9/16 E-5',
        '2/16 ---',
        '2/16 G-5',

        '3/16 F#5',
        '1/16 C#5',
        '2/16 ---',
        '4/16 A-5',
        '2/16 G-5',
        '2/16 F#5',
        '2/16 E#5',

        '6/16 F#5',
        '2/16 E-5',
        '4/16 D#5',
        '1/20 E-5',  # \
        '1/20 F#5',  # |
        '1/20 G-5',  #  > quintuplet (5 * 1/20 == 1/4 == 4/16)
        '1/20 A-5',  # |
        '1/20 A#5',  # /
    ], tempo=136, detune_cents=-2)
    # Currently, the synthesized sound is too regular (i.e. not very realistic).
    # This will result in audible peculiarities for some unlucky note and tuning
    # combinations. Hence the slight detuning.

    device_evaluator = AudioDeviceEvaluator(target_channels=2, min_channels=1, max_channels=4, target_latency=0.004)
    device_list = sd.query_devices()

    for device in sorted(device_list, key=device_evaluator.key_func, reverse=True):
        weight = device_evaluator.key_func(device)[0]
        if weight < 0:
            continue

        print(f"{weight:<7.3g}  lat: {device['default_low_output_latency']:<7.3g}  ch: {device['max_output_channels']:3d} \t[{sd.query_hostapis(device['hostapi'])['name']}]\t{device['name']}")

    with sd.OutputStream(samplerate=SAMPLE_RATE, device=max(sd.query_devices(), key=device_evaluator.key_func)['index'], channels=2, dtype=np.float32, latency=0.004) as stream:
        print(f"samplerate = {stream.samplerate}")
        print(f"channels = {stream.channels}")
        print(f"dtype = {stream.dtype}")
        print(f"samplesize = {stream.samplesize}")
        print(f"latency = {stream.latency}")
        print(f"blocksize = {stream.blocksize}")
        print(f"device = {device_list[stream.device]['name']}")
        print(f"hostapi = {sd.query_hostapis(device_list[stream.device]['hostapi'])['name']}")

        if stream.channels > 1:
            song = np.tile(song[:, np.newaxis], (1, stream.channels))

        pos = 0
        while pos < len(song):
            while not (n := stream.write_available):
                sleep(stream.latency / 4)
            stream.write(song[pos : pos + n])
            pos += n

        stream.write(np.zeros((2 * max(math.ceil(0.5 * SAMPLE_RATE), stream.blocksize), stream.channels), dtype=np.float32))


if __name__ == '__main__':
    main()
