from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import cv2


# Configure these globals before running the script.
input = "input.mp4"
folder = "frames"
discarded = "discarded"
prefixed = "frame_"
min_interval = 1.0
max_interval = 5.0
threshold_min = 0.80
threshold_max = 0.98


def video_duration(filename: str) -> float:
	result = subprocess.run(
		[
			"ffprobe", "-v", "error", "-show_entries", "format=duration",
			"-of", "default=noprint_wrappers=1:nokey=1", filename,
		],
		check=True,
		capture_output=True,
		text=True,
	)
	return float(result.stdout.strip())


def extract_frame(filename: str, timestamp: float, output: Path) -> None:
	subprocess.run(
		[
			"ffmpeg", "-y", "-ss", str(timestamp), "-i", filename,
			"-frames:v", "1", "-q:v", "2", str(output),
		],
		check=True,
		stdout=subprocess.DEVNULL,
		stderr=subprocess.DEVNULL,
	)


def ssim(first, second) -> float:
	# SSIM is computed on grayscale images with an 11x11 Gaussian window.
	first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
	second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
	c1, c2 = 6.5025, 58.5225
	first = first.astype("float64")
	second = second.astype("float64")
	mean1 = cv2.GaussianBlur(first, (11, 11), 1.5)
	mean2 = cv2.GaussianBlur(second, (11, 11), 1.5)
	variance1 = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mean1 * mean1
	variance2 = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mean2 * mean2
	covariance = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mean1 * mean2
	value = ((2 * mean1 * mean2 + c1) * (2 * covariance + c2)) / (
		(mean1 * mean1 + mean2 * mean2 + c1) * (variance1 + variance2 + c2)
	)
	return float(value.mean())


def main() -> None:
	output_dir = Path(folder)
	discarded_dir = Path(discarded)
	output_dir.mkdir(parents=True, exist_ok=True)
	discarded_dir.mkdir(parents=True, exist_ok=True)

	duration = video_duration(input)
	timestamp = 0.0
	previous = None
	index = 0

	while timestamp < duration:
		frame_path = output_dir / f"{prefixed}{index:06d}.jpg"
		extract_frame(input, timestamp, frame_path)
		current = cv2.imread(str(frame_path))
		if current is None:
			frame_path.unlink(missing_ok=True)
			break

		if previous is not None:
			if previous.shape != current.shape:
				current = cv2.resize(current, (previous.shape[1], previous.shape[0]))
			similarity = ssim(previous, current)
			# Discard frames outside the accepted SSIM range.
			if similarity < threshold_min or similarity > threshold_max:
				shutil.move(str(frame_path), discarded_dir / frame_path.name)

		previous = current
		index += 1
		timestamp += random.uniform(min_interval, max_interval)


if __name__ == "__main__":
	main()
