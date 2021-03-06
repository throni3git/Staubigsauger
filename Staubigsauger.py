import sys
import json
from pathlib import Path

from PySide2.QtCore import Qt, Slot, Signal
from PySide2.QtGui import QIcon

from PySide2 import QtWidgets, QtCore

import numpy as np
import scipy.signal

from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT, FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt

import soundfile

_default_config = {
    "filename_A": None,
    "filename_B": None,
    "NFFT": 2048,
    "n_std_mag": 1.5,
    "percentage_quiet_magnitudes": 20
}

_config = _default_config.copy()


def get_config(key: str):
    if key in _config:
        return _config[key]
    if key in _default_config:
        return _default_config[key]

    raise KeyError(f"Key {key} not found")


def set_config(key: str, val) -> None:
    if key not in _default_config:
        raise KeyError(f"Key {key} not found in _default_config")
    _config[key] = val


def load_config() -> None:
    path = Path(__file__).parent / "config.json"
    if path.exists():
        with open(str(path), "r") as f:
            global _config
            _config = json.load(f)
    else:
        print("Staubigsauger: use default settings")


def write_config() -> None:
    path = Path(__file__).parent / "config.json"
    with open(str(path), "w") as f:
        global _config
        json.dump(_config, f, indent=2)


class WidgetSignal(QtWidgets.QWidget):

    calculationByThresholdDesired = Signal()
    signalFilenameChanged = Signal(str)

    def __init__(self, title: str, parent=None, can_open: bool = True, can_save: bool = True, can_calculate: bool = True):
        super(WidgetSignal, self).__init__(parent)

        self.sig = np.zeros((2, 1024))
        self.fn_signal = ""
        self.fs = 48000
        self._percentage_quiet_magnitudes = get_config("percentage_quiet_magnitudes")
        self._std_mags = get_config("n_std_mag")

        _layout = QtWidgets.QVBoxLayout(self)

        # spectrogram view
        self.figure = Figure(figsize=(5, 5))
        self.figure.set_tight_layout(dict(pad=0.3))
        self.axes: plt.Axes = self.figure.add_subplot(111)

        self.canvas = FigureCanvasQTAgg(self.figure)
        _layout.addWidget(self.canvas)
        _layout.setStretchFactor(self.canvas, 1)
        _layout.addWidget(NavigationToolbar2QT(self.canvas, self))

        # file name
        self._group_filename = QtWidgets.QGroupBox(title)
        _layout.addWidget(self._group_filename)

        _layout_filename = QtWidgets.QVBoxLayout(self._group_filename)
        self._label_filename = QtWidgets.QLabel("Dateiname?")
        _layout_filename.addWidget(self._label_filename)

        _layout_filename_buttons = QtWidgets.QHBoxLayout()

        if can_calculate:
            _layout_parameters = QtWidgets.QVBoxLayout()

            # slider for adjusting the amount of samples in the quiet area used for threshold estimation
            _layout_param1 = QtWidgets.QHBoxLayout()
            self._slider_percent_quiet_mags = QtWidgets.QSlider(Qt.Horizontal)
            self._slider_percent_quiet_mags.setMinimum(1)
            self._slider_percent_quiet_mags.setMaximum(50)
            self._slider_percent_quiet_mags.valueChanged[int].connect(self._slider_percent_quiet_mags_changed)
            _layout_param1.addWidget(self._slider_percent_quiet_mags)

            self._slider_percent_quiet_mags_label = QtWidgets.QLabel()
            _layout_param1.addWidget(self._slider_percent_quiet_mags_label)

            self._slider_percent_quiet_mags.setValue(self._percentage_quiet_magnitudes)

            self._calc_button = QtWidgets.QPushButton("Calculate")
            self._calc_button.clicked.connect(self.calculationByThresholdDesired.emit)
            _layout_param1.addWidget(self._calc_button)

            # slider for adjusting the standard deviation used for threshold estimation
            _layout_param2 = QtWidgets.QHBoxLayout()
            self._slider_std_mags = QtWidgets.QSlider(Qt.Horizontal)
            self._slider_std_mags.setMinimum(1)
            self._slider_std_mags.setMaximum(400)
            self._slider_std_mags.valueChanged[int].connect(self._slider_std_mags_changed)
            _layout_param2.addWidget(self._slider_std_mags)

            self._slider_std_mags_label = QtWidgets.QLabel()
            _layout_param2.addWidget(self._slider_std_mags_label)

            self._slider_std_mags.setValue(int(self._std_mags*100))

            self._calc_button = QtWidgets.QPushButton("Calculate")
            self._calc_button.clicked.connect(self.calculationByThresholdDesired.emit)
            _layout_param2.addWidget(self._calc_button)

            _layout_parameters.addLayout(_layout_param1)
            _layout_parameters.addLayout(_layout_param2)
            _layout_filename_buttons.addLayout(_layout_parameters)

        _layout_filename_buttons.addStretch()

        if can_open:
            self._open_button = QtWidgets.QPushButton("Open")
            self._open_button.clicked.connect(self._on_filedialog_open_file)
            _layout_filename_buttons.addWidget(self._open_button)

        if can_save:
            self._save_button = QtWidgets.QPushButton("Save")
            self._save_button.clicked.connect(self._on_filedialog_save_file)
            _layout_filename_buttons.addWidget(self._save_button)

        _layout_filename.addLayout(_layout_filename_buttons)

    def _slider_percent_quiet_mags_changed(self, val):
        self._percentage_quiet_magnitudes = val
        self._slider_percent_quiet_mags_label.setText(f"{val} of the quiet magnitudes")

    def _slider_std_mags_changed(self, val):
        val /= 100
        self._std_mags = val
        self._slider_std_mags_label.setText(f"{val:.2f} times the std deviation")

    def set_data(self, data):
        self.sig = data.copy()
        self.sig = np.atleast_2d(self.sig)
        self.update_plot()

    def update_plot(self):
        nfft = get_config("NFFT")
        f, t, STFT = scipy.signal.stft(self.sig, self.fs, window='hann', nperseg=nfft)
        mag = np.abs(STFT)
        floor = 1e-12
        mag[mag < floor] = floor
        mag_log = 20*np.log10(mag)
        vmin = 20*np.log10(floor)
        vmax = 0

        self.axes.imshow(mag_log[0],
                         aspect="auto",
                         origin="lower",
                         cmap="jet",
                         interpolation="nearest",
                         vmin=vmin,
                         vmax=vmax)
        self.axes.grid(True, alpha=0.3)

        self.axes.set_xlabel("time in s")
        self.axes.set_xticks(np.arange(0, t[-1], 0.5) * self.fs * 2 / nfft)
        self.axes.set_xticklabels(np.arange(0, t[-1], 0.5))

        self.axes.set_ylabel("freq in kHz")
        self.axes.set_yticks(np.arange(0, f[-1], 1000) * nfft / self.fs)
        self.axes.set_yticklabels(np.arange(0, f[-1], 1000, dtype=np.int)//1000)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def open_file(self, fn_signal: str):
        self.fn_signal = fn_signal
        self._label_filename.setText(fn_signal)
        data, fs = soundfile.read(fn_signal, always_2d=True)
        self.fs = fs
        self.set_data(data.T)
        self.signalFilenameChanged.emit(fn_signal)

    def _on_filedialog_open_file(self):
        pn_before = Path(self.fn_signal if self.fn_signal is not None else __file__).parent

        fn_signal, _ = QtWidgets.QFileDialog.getOpenFileName(filter="*.wav", dir=str(pn_before))
        if fn_signal is not "":
            self._label_filename.setText(fn_signal)
            self.open_file(fn_signal)

    def save_file(self, fn_signal: str):
        self.fn_signal = fn_signal
        soundfile.write(self.fn_signal, self.sig.T, self.fs, subtype="PCM_32")
        self.signalFilenameChanged.emit(self.fn_signal)

    def _on_filedialog_save_file(self):
        pn_before = Path(self.fn_signal if self.fn_signal is not None else __file__).parent

        fn_signal, _ = QtWidgets.QFileDialog.getSaveFileName(filter="*.wav", dir=str(pn_before))
        if fn_signal is "":
            return
        if not fn_signal.endswith(".wav"):
            fn_signal = fn_signal + ".wav"
        if fn_signal is not "":
            self._label_filename.setText(fn_signal)
            self.save_file(fn_signal)


class MainStaubigsauger(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Staubigsauger")
        app_icon = QIcon("assets/icon.png")
        self.setWindowIcon(app_icon)

        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main)

        layout = QtWidgets.QHBoxLayout(self._main)
        self.widgetSignalA = WidgetSignal("Signal A", can_save=False, can_calculate=False)
        self.widgetSignalA.signalFilenameChanged.connect(self.signalFilenameChangedA)
        layout.addWidget(self.widgetSignalA)

        _label_convolution = QtWidgets.QLabel("<h1>=></h1>")
        layout.addWidget(_label_convolution)

        self.widgetSignalB = WidgetSignal("Signal B", can_open=False)
        self.widgetSignalB.signalFilenameChanged.connect(self.signalFilenameChangedB)
        self.widgetSignalB.calculationByThresholdDesired.connect(self.calculation_by_threshold)
        layout.addWidget(self.widgetSignalB)

        # initial IR file loading
        fn_wave = get_config("filename_A")
        if fn_wave == None:
            fn_wave = str(Path(__file__).parent / "example/A.wav")
        self.widgetSignalA.open_file(fn_wave)

        fn_wave = get_config("filename_B")
        if fn_wave == None:
            fn_wave = str(Path(__file__).parent / "example/B.wav")
        self.widgetSignalB.open_file(fn_wave)

    def signalFilenameChangedA(self, fn_out: str):
        set_config("filename_A", fn_out)
        write_config()

    def signalFilenameChangedB(self, fn_out: str):
        set_config("filename_B", fn_out)
        write_config()

    def calculation_by_threshold(self):

        # attenuate noise
        sig = self.widgetSignalA.sig
        fs = self.widgetSignalA.fs
        NFFT = get_config("NFFT")
        n_std_mag = self.widgetSignalB._std_mags
        percentage_quiet_magnitudes = self.widgetSignalB._percentage_quiet_magnitudes / 100

        f, t, Sxx_rec = scipy.signal.stft(sig, fs, window='hann', nperseg=NFFT)
        Sxx_rec = np.atleast_3d(Sxx_rec)
        mag_rec_db = 20 * np.log10(np.abs(Sxx_rec) + 1e-10)

        mask = np.ones_like(Sxx_rec)
        sf_len = 5
        # smoothing_filter = 1-np.abs(np.linspace(-1+2/sf_len, 1-2/sf_len, sf_len))
        smoothing_filter = 0.5 + 0.5*np.cos(np.linspace(-1, 1, sf_len+2)*np.pi)
        smoothing_filter = smoothing_filter[1:-2]
        smoothing_filter = np.outer(smoothing_filter, smoothing_filter)
        smoothing_filter /= np.sum(smoothing_filter)

        # smoove input signal spectrogram a little bit
        # mag_rec_db[:, :, 1:] = mag_rec_db[:, :, 1:]/2 + mag_rec_db[:, :, :-1]/2
        # mag_rec_db[:, :-1, :] = mag_rec_db[:, :-1, :]/2 + mag_rec_db[:, 1:, :]/2

        num_chan = mag_rec_db.shape[0]
        for chan in range(num_chan):
            # nullen aus der wertung nehmen
            the_max = np.max(mag_rec_db[chan])
            invalid = mag_rec_db[chan] <= -200
            mag_rec_db[chan, invalid] = the_max

            # use blocks where the energy overall is smallest
            E = np.mean(mag_rec_db[chan], axis=0)
            E_sorted = np.argsort(E)
            E_sorted_shrinked = E_sorted[:int(E_sorted.shape[0] * percentage_quiet_magnitudes)]
            shrinked_mag_db = mag_rec_db[chan, :, E_sorted_shrinked].T

            sorted_mag = np.sort(mag_rec_db[chan], axis=1)
            max_len_quiet_magnitudes = int(sorted_mag.shape[1] * percentage_quiet_magnitudes)
            max_len_quiet_magnitudes = max(1, max_len_quiet_magnitudes)
            # shrinked_mag_db = sorted_mag[:, :max_len_quiet_magnitudes]
            # shrinked_mag = shrinked_mag[:, shrinked_mag.shape[1]//10:]

            med_mag = np.median(shrinked_mag_db, axis=1)
            print("Statistics")
            print(f"  MEDIAN: {np.mean(med_mag)}")

            # attempt like in https://timsainburg.com/noise-reduction-python.html
            mean_mag_db = np.mean(shrinked_mag_db, axis=1)
            std_mag_db = np.std(shrinked_mag_db, axis=1)
            noise_floor_db = mean_mag_db + std_mag_db * n_std_mag

            print(f"  Mean: {np.mean(mean_mag_db)}")
            print(f"  Std: {np.mean(std_mag_db)}")
            print(f"  Noise: {np.mean(noise_floor_db)}")

            tiled = np.tile(noise_floor_db, (Sxx_rec.shape[-1], 1)).T
            indexed = mag_rec_db[chan] < tiled

            mask[chan, indexed] = 0
            mask[chan] = scipy.signal.fftconvolve(mask[chan], smoothing_filter, mode="same")
            Sxx_rec[chan] *= mask[chan]

        # plt.plot(np.log10(erg + 1e-12)*20)
        # plt.show()

        t_synth, x_synth = scipy.signal.istft(Sxx_rec, fs, window='hann', nperseg=NFFT)
        x_synth = x_synth[:, :sig.shape[1]]

        self.widgetSignalB.set_data(x_synth)


if __name__ == "__main__":

    load_config()

    qapp = QtWidgets.QApplication(sys.argv)

    main = MainStaubigsauger()
    main.show()

    sys.exit(qapp.exec_())
