# Driving Lab

Train an offline steering estimator on labelled frames from driving footage. A small ridge regression model predicts steering in [-1,1].

## Quick start

```sh
python -m pip install -r requirements.txt
python evaluate.py
python driving.py train demo/train.csv model.json
python driving.py predict model.json demo/test-00.png
```

## Using your own material

Use frames from your own clips and record frame,steer in CSV. Split by recording to evaluate a new camera or route. The bundled road drawings exercise the pipeline and do not measure driving skill in GTA VI. Predictions are offline.

Independent fan project. Grand Theft Auto is a trademark of its respective owners.
