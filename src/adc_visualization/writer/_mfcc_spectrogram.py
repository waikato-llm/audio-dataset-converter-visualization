import argparse
import os
from typing import List

import librosa
import librosa.display
import matplotlib.pyplot as plt
from wai.logging import LOGGING_WARNING

from adc.api import AudioData, SplittableStreamWriter, make_list
from seppl.placeholders import placeholder_list, InputBasedPlaceholderSupporter
from ._output_types import OUTPUT_TYPES, OUTPUT_TYPE_PNG


class MFCCSpectrogram(SplittableStreamWriter, InputBasedPlaceholderSupporter):

    def __init__(self, num_mfcc: int = None, dct_type: int = None, norm: str = None, lifter: int = None,
                 num_fft: int = None, hop_length: int = None, win_length: int = None,
                 window: str = None, center: bool = None, pad_mode: str = None,
                 power: float = None, cmap: str = None, dpi: int = None,
                 output_dir: str = None, output_type: str = OUTPUT_TYPE_PNG,
                 split_names: List[str] = None, split_ratios: List[int] = None,
                 logger_name: str = None, logging_level: str = LOGGING_WARNING):
        """
        Initializes the writer.

        :param num_mfcc: the number of MFCCs to return
        :type num_mfcc: int
        :param dct_type: the Discrete cosine transform (DCT) type (1|2|3). By default, DCT type-2 is used
        :type dct_type: int
        :param norm: If dct_type is 2 or 3, setting norm='ortho' uses an ortho-normal DCT basis. Normalization is not supported for dct_type=1. (options: none|ortho)
        :type norm: str
        :param lifter: If lifter>0, apply liftering (cepstral filtering) to the MFCC: M[n, :] <- M[n, :] * (1 + sin(pi * (n + 1) / lifter) * lifter / 2)
        :type lifter: int
        :param num_fft: the length of the windowed signal after padding with zeros. should be power of two
        :type num_fft: int
        :param hop_length: number of audio samples between adjacent STFT columns
        :type hop_length: int
        :param win_length: each frame of audio is windowed by window of length win_length and then padded with zeros to match num_fft. defaults to win_length = num_fft
        :type win_length: int
        :param window: a window function, such as scipy.signal.windows.hann
        :type window: str
        :param center: for centering the signal
        :type center: bool
        :param pad_mode: used when 'centering'
        :type pad_mode: str
        :param power: exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power, etc.
        :type power: float
        :param cmap: the Matplotlib colormap to use (append _r for reverse), automatically infers map if not provided; use 'gray_r' for grayscale; for available maps see: https://matplotlib.org/stable/gallery/color/colormap_reference.html
        :type cmap: str
        :param dpi: the dots per inch
        :type dpi: int
        :param output_dir: the output directory to save the audio/report in
        :type output_dir: str
        :param output_type: what type of image to generate
        :type output_type: str
        :param split_names: the names of the splits, no splitting if None
        :type split_names: list
        :param split_ratios: the integer ratios of the splits (must sum up to 100)
        :type split_ratios: list
        :param logger_name: the name to use for the logger
        :type logger_name: str
        :param logging_level: the logging level to use
        :type logging_level: str
        """
        super().__init__(split_names=split_names, split_ratios=split_ratios, logger_name=logger_name, logging_level=logging_level)
        self.num_mfcc = num_mfcc
        self.dct_type = dct_type
        self.norm = norm
        self.lifter = lifter
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode
        self.power = power
        self.cmap = cmap
        self.dpi = dpi
        self.output_dir = output_dir
        self.output_type = output_type

    def name(self) -> str:
        """
        Returns the name of the handler, used as sub-command.

        :return: the name
        :rtype: str
        """
        return "to-mfcc-spectrogram"

    def description(self) -> str:
        """
        Returns a description of the writer.

        :return: the description
        :rtype: str
        """
        return "Generates a plot from Mel-frequency cepstral coefficients."

    def _create_argparser(self) -> argparse.ArgumentParser:
        """
        Creates an argument parser. Derived classes need to fill in the options.

        :return: the parser
        :rtype: argparse.ArgumentParser
        """
        parser = super()._create_argparser()
        parser.add_argument("--num_mfcc", type=int, help="The number of MFCCs to return.", required=False, default=20)
        parser.add_argument("--dct_type", type=int, help="The Discrete cosine transform (DCT) type (1|2|3). By default, DCT type-2 is used.", required=False, default=2)
        parser.add_argument("--norm", type=str, help="If dct_type is 2 or 3, setting norm='ortho' uses an ortho-normal DCT basis. Normalization is not supported for dct_type=1. (options: none|ortho)", required=False, default="ortho")
        parser.add_argument("--lifter", type=int, help="If lifter>0, apply liftering (cepstral filtering) to the MFCC: M[n, :] <- M[n, :] * (1 + sin(pi * (n + 1) / lifter) * lifter / 2)", required=False, default=0)
        parser.add_argument("--num_fft", type=int, help="The length of the windowed signal after padding with zeros. should be power of two.", required=False, default=2048)
        parser.add_argument("--hop_length", type=int, help="The number of audio samples between adjacent STFT columns.", required=False, default=512)
        parser.add_argument("--win_length", type=int, help="The each frame of audio is windowed by window of length win_length and then padded with zeros to match num_fft. defaults to win_length = num_fft.", required=False, default=None)
        parser.add_argument("--window", type=str, help="A window function, such as scipy.signal.windows.hann.", required=False, default="hann")
        parser.add_argument("--center", action="store_true", help="For centering the signal.", required=False)
        parser.add_argument("--pad_mode", type=str, help="Used when 'centering'", required=False, default="constant")
        parser.add_argument("--power", type=float, help="The exponent for the magnitude melspectrogram. e.g., 1 for energy, 2 for power, etc.", required=False, default=2.0)
        parser.add_argument("--cmap", type=str, help="Matplotlib colormap to use (append _r for reverse), automatically infers map if not provided; use 'gray_r' for grayscale; for available maps see: https://matplotlib.org/stable/gallery/color/colormap_reference.html", required=False, default=None)
        parser.add_argument("--dpi", type=int, help="The dots per inch.", required=False, default=100)
        parser.add_argument("-o", "--output_dir", type=str, help="The directory to store the audio files in. Any defined splits get added beneath there. " + placeholder_list(obj=self), required=True)
        parser.add_argument("-t", "--output_type", choices=OUTPUT_TYPES, help="The type of image to geneate.", required=False, default=OUTPUT_TYPE_PNG)
        return parser

    def _apply_args(self, ns: argparse.Namespace):
        """
        Initializes the object with the arguments of the parsed namespace.

        :param ns: the parsed arguments
        :type ns: argparse.Namespace
        """
        super()._apply_args(ns)
        self.num_mfcc = ns.num_mfcc
        self.dct_type = ns.dct_type
        self.norm = ns.norm
        self.lifter = ns.lifter
        self.num_fft = ns.num_fft
        self.hop_length = ns.hop_length
        self.win_length = ns.win_length
        self.window = ns.window
        self.center = ns.center
        self.pad_mode = ns.pad_mode
        self.power = ns.power
        self.cmap = ns.cmap
        self.dpi = ns.dpi
        self.output_dir = ns.output_dir
        self.output_type = ns.output_type

    def accepts(self) -> List:
        """
        Returns the list of classes that are accepted.

        :return: the list of classes
        :rtype: list
        """
        return [AudioData]

    def initialize(self):
        """
        Initializes the processing, e.g., for opening files or databases.
        """
        super().initialize()
        if self.num_mfcc is None:
            self.num_mfcc = 20
        if self.dct_type is None:
            self.dct_type = 2
        if self.norm is None:
            self.norm = "ortho"
        if self.lifter is None:
            self.lifter = 0
        if self.num_fft is None:
            self.num_fft = 2048
        if self.hop_length is None:
            self.hop_length = 512
        if self.window is None:
            self.window = "hann"
        if self.center is None:
            self.center = False
        if self.pad_mode is None:
            self.pad_mode = "constant"
        if self.power is None:
            self.power = 2.0

    def write_stream(self, data):
        """
        Saves the data one by one.

        :param data: the data to write (single record or iterable of records)
        """
        for item in make_list(data):
            sub_dir = self.session.expand_placeholders(self.output_dir)
            if self.splitter is not None:
                split = self.splitter.next()
                sub_dir = os.path.join(sub_dir, split)
            if not os.path.exists(sub_dir):
                self.logger().info("Creating dir: %s" % sub_dir)
                os.makedirs(sub_dir)

            audio = item.audio
            if not item.is_mono:
                audio = librosa.to_mono(audio)

            # generate spectrogram
            mfccs = librosa.feature.mfcc(y=audio, sr=item.sample_rate, n_mfcc=self.num_mfcc, dct_type=self.dct_type,
                                         norm=(None if (self.norm == "none") else self.norm), lifter=self.lifter,
                                         n_fft=self.num_fft, hop_length=self.hop_length,
                                         win_length=self.win_length, window=self.window, center=self.center,
                                         pad_mode=self.pad_mode, power=self.power)

            # plot
            fig, ax = plt.subplots()
            if self.cmap is not None:
                librosa.display.specshow(mfccs, x_axis='time', y_axis='linear', ax=ax, cmap=self.cmap)
            else:
                librosa.display.specshow(mfccs, x_axis='time', y_axis='linear', ax=ax)
            path = os.path.join(sub_dir, os.path.splitext(item.audio_name)[0] + "." + self.output_type)
            self.logger().info("Writing plot to: %s" % path)
            plt.axis('off')
            plt.savefig(path, format=self.output_type, bbox_inches='tight', pad_inches=0, dpi=self.dpi)
            plt.close('all')
