## **Getting Started with YOLO26 MLX**

A 10-minute guide to get YOLO26 MLX running on your Mac.

### **What is YOLO26 MLX?**

YOLO26 MLX is webAI's pure [MLX](https://github.com/ml-explore/mlx) implementation of YOLO26 for Apple Silicon. It runs entirely on-device with Metal GPU acceleration. No PyTorch dependency at runtime, no cloud calls, no waiting on someone else's API.

The repo: [github.com/thewebAI/yolo-mlx](https://github.com/thewebAI/yolo-mlx)

### **Requirements**

* Apple Silicon Mac (M1, M2, M3, or M4)  
* macOS recent enough to support MLX (14.0+ recommended)  
* Python 3.10 or newer  
* \~2GB of free disk space for models and dependencies

If you're on an Intel Mac, this won't work. The whole point of MLX is Apple Silicon native acceleration.

### **5-minute setup**

Clone the repo, create a virtual environment, install the package, and download a model.

bash

```shell
git clone https://github.com/thewebAI/yolo-mlx.git
cd yolo-mlx
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[convert]"
```

Download a pretrained model and convert it to MLX format. We recommend starting with `yolo26n` (the smallest model, \~170 FPS on M4 Pro) for fast iteration:

bash

```shell
bash scripts/download_yolo26_models.sh
yolo26 converters convert models/yolo26n.pt -o models/yolo26n.npz --verify
```

You now have a working MLX model at `models/yolo26n.npz`.

### **Run your first detection**

Grab a sample image and run inference:

bash

```shell
mkdir -p images
curl -L -o images/bus.jpg https://ultralytics.com/images/bus.jpg
```

Then in Python:

python

```py
from yolo26mlx import YOLO

model = YOLO("models/yolo26n.npz")
results = model.predict("images/bus.jpg", conf=0.25)
print(results[0])
results[0].save()
```

The output image with bounding boxes will save to the `results/` directory. That's your hello world.

### **Running on your webcam**

Most builds for this hackathon will use live camera input. Here's a minimal webcam loop:

python

```py
import cv2
from yolo26mlx import YOLO

model = YOLO("models/yolo26n.npz")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.25)
    annotated = results[0].plot()

    cv2.imshow("YOLO26 MLX", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

You'll need `opencv-python` installed: `pip install opencv-python`.

The first time you run this, macOS will ask for camera permission for your terminal or IDE. Grant it. If you skip the prompt by accident, go to System Settings → Privacy & Security → Camera and enable it for whatever app is running Python.

### **Model size vs. speed**

Five models, pick what fits your use case:

| Model | mAP | FPS (M4 Pro) | When to use |
| ----- | ----- | ----- | ----- |
| yolo26n | 40.2% | 170 | Real-time, low-latency demos |
| yolo26s | 47.6% | 105 | Balanced default |
| yolo26m | 52.3% | 55 | Higher accuracy, still fast |
| yolo26l | 53.9% | 44 | Accuracy over speed |
| yolo26x | 56.7% | 24 | Max accuracy, slower |

For most hackathon demos, `yolo26n` or `yolo26s` will be the right call. Real-time webcam apps almost always want `n`.

### **Common gotchas**

**MLX won't install or import.** Make sure you're on Apple Silicon. `python3 -c "import platform; print(platform.processor())"` should return `arm`. If it returns `i386`, you're on Intel and this won't work.

**Camera permission isn't granted.** macOS blocks camera access by default. The first time your Python script accesses the camera, you'll get a system prompt. Approve it. If you missed it, go to System Settings → Privacy & Security → Camera and toggle on the app you're using.

**Model conversion fails.** The `yolo26 converters convert` step needs the `[convert]` extras installed. If you skipped `pip install -e ".[convert]"`, go back and run it.

**Inference is slow.** Make sure you convert the `.pt` weights to `.npz`. If you're running inference on the raw `.pt` file you're using PyTorch under the hood, not MLX. The `.npz` is the MLX-native format.

**Out of memory on large models.** `yolo26x` needs more memory than `yolo26n`. If you're on an 8GB M1, stick with `n` or `s`.

### **Want to fine-tune?**

YOLO26 MLX supports full training and fine-tuning on Apple Silicon. See the [training guide](https://github.com/thewebAI/yolo-mlx/blob/main/GUIDE_TRAINING_BENCHMARK.md) in the repo. For most 5-day sprint submissions, pretrained inference is enough — fine-tuning is a heavier lift.

### **License note**

YOLO26 MLX is licensed under [AGPL-3.0](https://github.com/thewebAI/yolo-mlx/blob/main/LICENSE). What this means for the hackathon: you can absolutely use it, fork it, modify it, and submit it. If you ever plan to deploy your build as a hosted service or web app for real users, AGPL requires you to publish your source code under the same license. For demos, prototypes, and hackathon submissions, this isn't something to worry about.

### **Where to ask questions**

If you're stuck, head to the Hackathons category on community.webai.com. Mitch and Fatih on our ML team are watching the threads and will help unblock you. There's also a mid-sprint Q\&A scheduled — check the brief for the time.

Happy building.
