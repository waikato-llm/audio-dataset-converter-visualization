# to-mfcc-spectrogram

* accepts: adc.api.AudioData

Generates a plot from Mel-frequency cepstral coefficients.

```
usage: to-mfcc-spectrogram [-h] [-l {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
                           [-N LOGGER_NAME]
                           [--split_ratios SPLIT_RATIOS [SPLIT_RATIOS ...]]
                           [--split_names SPLIT_NAMES [SPLIT_NAMES ...]]
                           [--split_group SPLIT_GROUP] [--num_mfcc NUM_MFCC]
                           [--dct_type DCT_TYPE] [--norm NORM]
                           [--lifter LIFTER] [--num_fft NUM_FFT]
                           [--hop_length HOP_LENGTH] [--win_length WIN_LENGTH]
                           [--window WINDOW] [--center] [--pad_mode PAD_MODE]
                           [--power POWER] [--cmap CMAP] [--dpi DPI] -o
                           OUTPUT_DIR [-t {jpg,png}]

Generates a plot from Mel-frequency cepstral coefficients.

options:
  -h, --help            show this help message and exit
  -l {DEBUG,INFO,WARNING,ERROR,CRITICAL}, --logging_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        The logging level to use. (default: WARN)
  -N LOGGER_NAME, --logger_name LOGGER_NAME
                        The custom name to use for the logger, uses the plugin
                        name by default (default: None)
  --split_ratios SPLIT_RATIOS [SPLIT_RATIOS ...]
                        The split ratios to use for generating the splits
                        (must sum up to 100) (default: None)
  --split_names SPLIT_NAMES [SPLIT_NAMES ...]
                        The split names to use for the generated splits.
                        (default: None)
  --split_group SPLIT_GROUP
                        The regular expression with a single group used for
                        keeping items in the same split, e.g., for identifying
                        the base name of a file or the sample ID. (default:
                        None)
  --num_mfcc NUM_MFCC   The number of MFCCs to return. (default: 20)
  --dct_type DCT_TYPE   The Discrete cosine transform (DCT) type (1|2|3). By
                        default, DCT type-2 is used. (default: 2)
  --norm NORM           If dct_type is 2 or 3, setting norm='ortho' uses an
                        ortho-normal DCT basis. Normalization is not supported
                        for dct_type=1. (options: none|ortho) (default: ortho)
  --lifter LIFTER       If lifter>0, apply liftering (cepstral filtering) to
                        the MFCC: M[n, :] <- M[n, :] * (1 + sin(pi * (n + 1) /
                        lifter) * lifter / 2) (default: 0)
  --num_fft NUM_FFT     The length of the windowed signal after padding with
                        zeros. should be power of two. (default: 2048)
  --hop_length HOP_LENGTH
                        The number of audio samples between adjacent STFT
                        columns. (default: 512)
  --win_length WIN_LENGTH
                        The each frame of audio is windowed by window of
                        length win_length and then padded with zeros to match
                        num_fft. defaults to win_length = num_fft. (default:
                        None)
  --window WINDOW       A window function, such as scipy.signal.windows.hann.
                        (default: hann)
  --center              For centering the signal. (default: False)
  --pad_mode PAD_MODE   Used when 'centering' (default: constant)
  --power POWER         The exponent for the magnitude melspectrogram. e.g., 1
                        for energy, 2 for power, etc. (default: 2.0)
  --cmap CMAP           Matplotlib colormap to use (append _r for reverse),
                        automatically infers map if not provided; use 'gray_r'
                        for grayscale; for available maps see: https://matplot
                        lib.org/stable/gallery/color/colormap_reference.html
                        (default: None)
  --dpi DPI             The dots per inch. (default: 100)
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        The directory to store the audio files in. Any defined
                        splits get added beneath there. Supported
                        placeholders: {INPUT_PATH}, {INPUT_NAMEEXT},
                        {INPUT_NAMENOEXT}, {INPUT_EXT}, {INPUT_PARENT_PATH},
                        {INPUT_PARENT_NAME} (default: None)
  -t {jpg,png}, --output_type {jpg,png}
                        The type of image to geneate. (default: png)
```
