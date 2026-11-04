from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import av
import cv2
import numpy as np

# Configure these globals before running the script.
input = "input.mp4"
folder = "frames_6"
discarded = "discarded_6"
prefixed = "frame_"
min_interval = 4.0
max_interval = 13.0
threshold_min = 0.51
threshold_max = 0.8

C1: float = 6.5025
C2: float = 58.5225


def extract_frame(container: av.container.InputContainer, stream: av.video.stream.VideoStream, timestamp: float) -> np.ndarray | None:
	"""Seek to timestamp and decode frame using PyAV (supports AV1, H.264, etc.)."""
	target_pts = int(timestamp / stream.time_base)
	container.seek(target_pts, any_frame=False, backward=True, stream=stream)
	for frame in container.decode(stream):
		if frame.pts is not None and frame.pts >= target_pts:
			return frame.to_ndarray(format="bgr24")
	return None


def compute_frame_stats(frame_gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Compute mean and variance on a float grayscale image."""
	mean = cv2.GaussianBlur(frame_gray, (11, 11), 1.5)
	variance = cv2.GaussianBlur(frame_gray * frame_gray, (11, 11), 1.5) - mean * mean
	return mean, variance


def fast_ssim(
	first_gray: np.ndarray,
	mean1: np.ndarray,
	variance1: np.ndarray,
	second_gray: np.ndarray,
	mean2: np.ndarray,
	variance2: np.ndarray,
) -> float:
	covariance = cv2.GaussianBlur(first_gray * second_gray, (11, 11), 1.5) - mean1 * mean2
	value = ((2 * mean1 * mean2 + C1) * (2 * covariance + C2)) / (
		(mean1 * mean1 + mean2 * mean2 + C1) * (variance1 + variance2 + C2)
	)
	return float(value.mean())


def main() -> None:
	output_dir = Path(folder)
	discarded_dir = Path(discarded)
	output_dir.mkdir(parents=True, exist_ok=True)
	discarded_dir.mkdir(parents=True, exist_ok=True)

	container = av.open(input)
	stream = container.streams.video[0]
	duration = float(container.duration / 1_000_000) if container.duration else 0.0

	timestamp = 0.0
	index = 0

	first_gray = None
	ff_mean = None
	ff_variance = None

	prev_gray = None
	prev_mean = None
	prev_variance = None

	try:
		while timestamp < duration:
			current = extract_frame(container, stream, timestamp)
			if current is None:
				break

			frame_path = output_dir / f"{prefixed}{index:06d}.jpg"

			# Convert current to float32 grayscale and compute stats once
			curr_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
			curr_mean, curr_variance = compute_frame_stats(curr_gray)

			if index == 0:
				first_gray = curr_gray
				ff_mean = curr_mean
				ff_variance = curr_variance
				cv2.imwrite(str(output_dir / frame_path.name), current)
			else:
				if prev_gray.shape != curr_gray.shape:
					curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))
					curr_mean, curr_variance = compute_frame_stats(curr_gray)

				similarity = fast_ssim(
					prev_gray, prev_mean, prev_variance,
					curr_gray, curr_mean, curr_variance,
				)
				ff_similarity = fast_ssim(
					first_gray, ff_mean, ff_variance,
					curr_gray, curr_mean, curr_variance,
				)

				print(f"Frame {index:06d}: SSIM with previous = {similarity:.4f}, SSIM with first = {ff_similarity:.4f}")
				# Discard frames outside the accepted SSIM range
				if (
					(similarity < threshold_min or similarity > threshold_max) or 
					(ff_similarity < threshold_min or ff_similarity > threshold_max)
				):
					print(f"Frame {index:06d} discarded.")
					cv2.imwrite(str(discarded_dir / frame_path.name), current)
				else:
					cv2.imwrite(str(output_dir / frame_path.name), current)
				print("--------------------------------")

			prev_gray = curr_gray
			prev_mean = curr_mean
			prev_variance = curr_variance

			index += 1
			timestamp += random.uniform(min_interval, max_interval)
	finally:
		container.close()


if __name__ == "__main__":
	main()