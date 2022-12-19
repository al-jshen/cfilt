# cfilt
Current filtering for PIC simulations


## What all the different things here are for

### Plotting 

There is a [scripts/plot.py](https://github.com/al-jshen/cfilt/blob/master/scripts/plot.py) tool for generating grids of plots that can be useful for comparing different filters. Run this with `python plot.py`. There is again a `--help` flag. Several filters are implemented (e.g., Gaussian, wavelet, non-linear means, Fourier), and you can visualize different combinations of the original image, the filtered image, the filtered image in Fourier space, gradients, and more. 

### Notebooks

See [notebooks](https://github.com/al-jshen/cfilt/tree/master/notebooks) for a bunch of notebooks that play around with the different filters. 

### CNN

There is some code for training/applying a convolutional neural network. Look in [cfilt/utils.py](https://github.com/al-jshen/cfilt/blob/master/cfilt/utils.py) and [scripts/train_cnn.py](https://github.com/al-jshen/cfilt/blob/master/scripts/train_cnn.py). There is a tool to load in current files to stream the data to the neural network for training, and a script to train the neural network with different parameters. The network can take low ppc images and learn to map them to high ppc images. If you train and save the parameters of a neural network, it can also be used in the plotting script (see above). The things that mention "cascade" are for training multiple neural networks that try to do denoising in series (e.g., 4 -> 8 -> 16 -> 32 -> 64 instead of 4 -> 64).

### Rust code (`src`, `bin`, `Cargo.toml`) 

This is for taking the output files from Tristan and outputting some H5 files with the current. To run, install Rust with [rustup](https://rustup.rs/), build the binary with `cargo build --release`, and then run the binary (you can use `--help` to see the available options). 
