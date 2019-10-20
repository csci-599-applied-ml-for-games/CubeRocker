### Part 1 Detect Objects in Game
- Game plays on Flash Player
- ffmpeg capture streaming video from screen and send it to simulated device
  - Kernel Module v4l2loopback simulates a webcam
- Yolo v3 get input from simulated webcam and do inference and get result.