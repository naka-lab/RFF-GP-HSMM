# RFF-GP-HSMM


This is an implementation of RFF-GP-HSMM, which introduces Random Fourier Features into [GP-HSMM](https://github.com/naka-lab/GP-HSMM) to enable fast and scalable segmentation of time series data. For details, please refer to the paper below:

```
@article{saito2025scalable,
  title={Scalable Unsupervised Segmentation via Random Fourier Feature-based GP-HSMM},
  author={Issei Saito, Masatoshi Nagano, Tomoaki Nakamura, Daichi Mochihashi, Koki Mimura},
  journal={arXiv:2507.10632},
  year={2025},
}
```
[PDF](https://arxiv.org/abs/2507.10632)

## How to Run
```
python main.py
```

Programs written in Cython will be automatically compiled at runtime.
If you encounter compilation errors with the Visual Studio compiler on Windows, please edit:

```
(Python installation directory)/Lib/distutils/msvc9compiler.py
```

Inside the `get_build_version()` function, replace the following line:

```
majorVersion = int(s[:-2]) - 6
```

with the version number of the Visual Studio you wish to use.
For example, for VS2012, set:

```
majorVersion = 11
```


## Output Files

When executed, the following files and directories will be created in the specified folder:

| File Name| Description |
| ---- | --- |
| class{c}.npy         | A collection of segments classified into class c                                                              |
| class{c}\_dim{d}.png | Plot of the d-th dimension of segments classified into class c                                                |
| class{c}/gp{d}/      | Parameters of RFF and Bayesian linear regression for the d-th dimension of class c                            |
| segm{n}.txt          | Segmentation result of the n-th sequence. Column 1: segment class, Column 2: flag indicating segment boundary |
| trans\_bos.npy       | Probability that each class appears at the beginning of a sequence                                            |
| trans\_eos.npy       | Probability that each class appears at the end of a sequence                                                  |
| trans.npy            | Transition probabilities of each class appearing after a given class                                          |
