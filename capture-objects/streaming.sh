# some cmd for start streaming the game and begin objects detection
# init simulated webcam
sudo modprobe v4l2loopback
# get mouse location
xdotool getmouselocation
# ffmpeg capture screen and send to camera
ffmpeg -f x11grab -r 15 -s 798x510 -i :1+985,904 -vcodec rawvideo -pix_fmt yuv420p -threads 0 -f v4l2 /dev/video0
# view webcam content
vlc v4l2:///dev/video0