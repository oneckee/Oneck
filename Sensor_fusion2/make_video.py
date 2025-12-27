import subprocess

cmd = [
    "ffmpeg",
    "-y",
    "-framerate", "10",
    "-i", "output/Track_result/track_%06d.png",
    "-c:v", "libx264",
    "-pix_fmt", "yuv420p",
    "output/track_result.mp4"
]

subprocess.run(cmd, check=True)
print("✅ output/track_result.mp4 생성 완료")
