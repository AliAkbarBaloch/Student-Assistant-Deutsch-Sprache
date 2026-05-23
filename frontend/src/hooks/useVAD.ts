import { useRef, useCallback } from "react";

const SPEECH_THRESHOLD = 12;    // 0–255 amplitude — above = speech
const SILENCE_DURATION = 1500;  // ms of silence before auto-sending
const MIN_SPEECH_MS    = 400;   // ignore clips shorter than this

interface VADOptions {
  stream: MediaStream;
  onSpeechStart: () => void;
  onSpeechEnd: () => void;
}

/**
 * Voice Activity Detection hook.
 * Uses Web Audio API AnalyserNode to detect speech vs silence.
 * Returns start() and stop() functions.
 */
export function useVAD() {
  const audioCtxRef   = useRef<AudioContext | null>(null);
  const intervalRef   = useRef<number | null>(null);
  const speakingRef   = useRef(false);
  const silenceStart  = useRef<number | null>(null);

  const start = useCallback(({ stream, onSpeechStart, onSpeechEnd }: VADOptions) => {
    audioCtxRef.current = new AudioContext();
    const analyser      = audioCtxRef.current.createAnalyser();
    analyser.fftSize    = 256;
    audioCtxRef.current.createMediaStreamSource(stream).connect(analyser);

    const buf = new Uint8Array(analyser.frequencyBinCount);

    intervalRef.current = window.setInterval(() => {
      analyser.getByteFrequencyData(buf);
      const vol      = buf.reduce((a, b) => a + b, 0) / buf.length;
      const speaking = vol > SPEECH_THRESHOLD;

      if (speaking) {
        silenceStart.current = null;
        if (!speakingRef.current) {
          speakingRef.current = true;
          onSpeechStart();
        }
      } else if (speakingRef.current) {
        if (!silenceStart.current) {
          silenceStart.current = Date.now();
        } else if (Date.now() - silenceStart.current >= SILENCE_DURATION) {
          speakingRef.current  = false;
          silenceStart.current = null;
          onSpeechEnd();
        }
      }
    }, 80);
  }, []);

  const stop = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    audioCtxRef.current?.close();
    audioCtxRef.current  = null;
    speakingRef.current  = false;
    silenceStart.current = null;
  }, []);

  return { start, stop, MIN_SPEECH_MS };
}
