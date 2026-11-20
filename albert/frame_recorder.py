"""Raw head-camera recording during a run: every frame the detector gets, halved (1280x720 -> 640x360), streamed into an H.264
video by a separate, low-priority ffmpeg process, plus the frames' timestamps.

    <log>_camera.mp4          the frames, in order (constant 22 fps container rate; the real times are in the CSV)
    <log>_camera.csv          frame index, received_at (the run's clock), t_capture (received_at - camera latency)

The control loop only hands a frame over (a queue): halving and encoding run in this thread and in ffmpeg. When the encoder
cannot keep up, frames are dropped (counted), never the control loop slowed. tools/export_lerobot.py puts these frames in the
LeRobot dataset (observation.images.head). The frames show the room and the people in it: they stay local (gitignored).
"""
from __future__ import annotations

import csv
import os
import queue
import shutil
import subprocess
import threading


_OPEN = []


def close_all():
    """Finish every recording still open (the run ended on an error): a video is never left unreadable."""
    for r in list(_OPEN):
        print(r.close(), flush=True)


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


class FrameRecorder:
    def __init__(self, path_mp4, scale=2, fps=22, crf=23, max_queue=44):
        self.path, self.scale, self.fps, self.crf = path_mp4, scale, fps, crf
        self.q = queue.Queue(maxsize=max_queue)
        self.proc, self.size, self.rows = None, None, []
        self.n_dropped = self.n_written = 0
        self.error = None
        self.exe = _ffmpeg()
        self.closed = False
        _OPEN.append(self)
        self.thread = threading.Thread(target=self._run, daemon=True)
        if self.exe:
            self.thread.start()
        else:
            self.error = "no ffmpeg (pip install imageio-ffmpeg)"

    def add(self, img, received_at, t_capture):
        """From the detector thread: never blocks."""
        if self.error:
            return
        try:
            self.q.put_nowait((img, received_at, t_capture))
        except queue.Full:
            self.n_dropped += 1

    def _start(self, h, w):
        self.size = (w, h)
        cmd = [self.exe, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps),
               "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(self.crf), "-threads", "1",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", self.path]
        if os.name == "posix" and shutil.which("nice"):
            cmd = ["nice", "-n", "10"] + cmd  # the control loop first
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            img, received_at, t_capture = item
            try:
                small = img[::self.scale, ::self.scale, :3]
                h, w = small.shape[:2]
                h, w = h - h % 2, w - w % 2  # yuv420p needs even sizes
                if self.proc is None:
                    self._start(h, w)
                if (w, h) != self.size:  # a resolution change mid-run: skip those frames
                    self.n_dropped += 1
                    continue
                self.proc.stdin.write(small[:h, :w].tobytes())
                self.rows.append([self.n_written, received_at, t_capture])
                self.n_written += 1
            except (BrokenPipeError, OSError) as exc:
                self.error = repr(exc)
                return

    def close(self):
        """Finish the video and write the timestamps; returns a one-line summary."""
        if self.closed:
            return "camera recording: already closed"
        self.closed = True
        if self in _OPEN:
            _OPEN.remove(self)
        if self.thread.is_alive():
            self.q.put(None)
            self.thread.join(timeout=20)
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            self.proc.wait(timeout=30)
        if not self.rows:
            return f"camera recording: nothing written{' (' + self.error + ')' if self.error else ''}"
        with open(self.path[:-4] + ".csv", "w", newline="") as fh:
            csv.writer(fh).writerows([["frame", "received_at", "t_capture"]] + self.rows)
        dur = self.rows[-1][1] - self.rows[0][1]
        return (f"camera recording: {self.path} ({self.n_written} frames, {self.size[0]}x{self.size[1]}, "
                f"{self.n_written / max(dur, 1e-6):.1f} fps, {self.n_dropped} dropped)")
