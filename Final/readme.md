# How to install
```bash
$ conda create -n cv_final python=3.12.3
$ conda activate cv_final
$ pip install -r requirement.txt
```

# How to execute
```bash
$ ./run.sh
```
`-s`: path to save solutions  

`-g`: path to ground truth
`-st`: start frame
`-end`: end frame

# How to evaluation
```bash
$ ./test.sh
```
`-s`: path to solution
`-g`: path to ground truth

# Directory architecture

```
|----- src
    |----- gt
        |----- 000.png
        |----- ...
        |----- 128.png

    |----- homography
        |----- 001_L.npy
        |----- 001_R.npy
        |----- ...
        |----- 127_L.npy
        |----- 127_R.npy

    |----- solution
        |----- 001.png, s_001.txt, m_001.txt, H1_001.npy, H2_001.npy
        |----- ...
        |----- 127.png, s_127.txt, m_127.txt, H1_127.npy, H2_127.npy

    |----- eval.py, functions.py, gen_frame.py

    |----- run.sh, test.sh

    |----- method.md, requirement.txt, result.txt

|----- CV24S_MediaTek.pdf, MediaTek_Supplement.pdf, processing_order.pdf, slide.pptx, readme.md
```
