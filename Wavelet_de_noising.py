import numpy as np
import pywt
import matplotlib.pyplot as plt


class Wavelet_de_noising:
    def __init__(self, signal, plot=False):
        self.signal = signal
        self.plot = plot

    def decompose(self, wavelet='db18', level=1):
        coeff = pywt.wavedec(self.signal, wavelet, mode="per")
        sigma = (1 / 0.6745) * self.madev(coeff[-level])
        uthresh = sigma * np.sqrt(2 * np.log(len(self.signal)))
        coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
        filtered_signal = pywt.waverec(coeff, wavelet, mode='per')
        if self.plot:
            self.plot_signal(filtered_signal, wavelet)
        return filtered_signal

    def plot_signal(self, filtered, wav):
        raw = np.array(self.signal)
        filtered = np.array(filtered)
        plt.figure(figsize=(10, 6))
        plt.plot(raw, label='Raw')
        plt.plot(filtered, label='Filtered')
        plt.legend()
        plt.title(f"DWT Denoising with {wav} Wavelet", size=15)
        plt.show()

    def get_wavelets(self):
        for wav in pywt.wavelist():
            print(wav)

    def madev(self, d, axis=None):
        """ Mean absolute deviation of a signal """
        return np.mean(np.absolute(d - np.mean(d, axis)), axis)
