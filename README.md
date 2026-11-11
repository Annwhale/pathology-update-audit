![Welcome](assets/welcome.gif)

# Pathology update audit

Code for *Quality-feature dependence and endpoint-dependent ranking in colorectal histology AI updates*.

## Run

Python 3.12. Install dependencies, then supply the accompanying research package:

```bash
pip install -r requirements.txt
python replay.py /path/to/research-package
```

The replay checks 36,288 patch, 69,120 colorectal ROI and 159,456 breast ROI records against saved probabilities and native reference counts.

Additional checks:

```bash
python check_probabilities.py /path/to/research-package
python check_pixels.py /path/to/research-package
python check_geometry.py /path/to/research-package
```

`fusion.py` contains the fusion-feature definitions, source-only head fitting, calibration, paired error accounting and spatial metrics. Dataset arrays, model weights and manuscript files are supplied separately.

## Data sources

- [TSR-CRC](https://doi.org/10.5281/zenodo.4024676)
- [BCSS](https://github.com/PathologyDataScience/BCSS)
- [Colorectal tissue cores](https://doi.org/10.18710/DIGQGQ)

Contact: zhangleyao@suda.edu.cn.
