create-env:
	./scripts/create-env.sh
download-models:
	./scripts/download_models.sh ./models.txt
post-process-demo:
	ffmpeg -i ./output/screen_capture.mp4 -vcodec libx264 -acodec aac ./output/screen_capture_h265.mp4
