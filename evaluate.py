from pathlib import Path
from tempfile import TemporaryDirectory
import csv, statistics
from driving import train,predict
demo=Path(__file__).parent/'demo'
if __name__=='__main__':
    with TemporaryDirectory() as work:
        model=Path(work)/'model.json';train(demo/'train.csv',model)
        with (demo/'test.csv').open() as f:
            errors=[abs(float(r['steer'])-predict(model,demo/r['frame'])) for r in csv.DictReader(f)]
        print(f'{len(errors)} held-out synthetic frames, mean absolute steering error: {statistics.mean(errors):.3f}')
