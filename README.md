# GTA VI Driving Lab

**How much steering can you infer from one frame?** Train a small regression model on labelled road images, then inspect its prediction on a frame it has never seen.

![One held-out road drawing used by the baseline](demo/test-00.png)

## Run the baseline

```sh
python -m pip install -r requirements.txt
python evaluate.py
```

The bundled split has **40 training drawings and 15 held-out drawings**. The measured mean absolute steering error is **0.020** on a scale from −1 to +1. This is a result on synthetic road shapes. Gameplay footage still needs a separate evaluation.

## Make one prediction

```sh
python driving.py train demo/train.csv model.json
python driving.py predict model.json demo/test-00.png
```

Supply a CSV with `frame,steer` for your own capture. The model crops the lower half of the frame, reduces it to a grayscale grid and fits a ridge regression. Keep each recording entirely in training or evaluation so neighbouring frames cannot leak across the split. Predictions run offline.

---

[Explore the GTA VI project collection](https://github.com/Berserk-hub150/gta6-resource-index) · Independent fan project. Grand Theft Auto is a trademark of its respective owners.
