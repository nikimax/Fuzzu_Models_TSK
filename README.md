# Hybrid fuzzy model for Stock returns forecasting
 We present a model for forecasting financial time series. It was created <br/>
 to forecast stock returns, but can be widely used in other areas. The model<br/>
 is based on the fuzzy system of Takagi, Sugeno and Kang, which provides<br/>
 a polynomial dependence with soft switching between equations. <br/>
 Vector autoregression (VAR) is used to obtain predictions of regressors.<br/>
 Wavelet decomposition is used as preprocessing and clears time series from noise.<br/>
 We also use FCM fuzzy clustering to split the data into fuzzy sets and make it ready for<br/>
 the TSK model use.

### model structure:
1. [Data de-noising with Wavelet decomposition](./Wavelet_de_noising.py)
2. [VAR forecast for factors](./VAR_estimation.R)
3. [Fuzzy clustering with FCM algorithm](./FCM_fuzzy_clustering.py)
4. [Takagi-Sugeno_Kang fuzzy system forecast](./Takagi_Sugeno_Kang.py)

## Wavelet decomposition
Financial time series have noise that interferes with the identification of trends.<br/>
Also, noise, outliers and heteroscedasticity make the time series weakly stationary.<br/>
Therefore, we use wavelet decomposition to de-noise. The main idea of this approach is<br/>
to decompose the signal into two parts, High-frequency (Wavelet) and Approximating. The<br/>
high-frequency part is then removed, thus the signal is cleared from noise<br/>

### Code documentation for Wavelets
#### Class
The wavelet decomposition is written as a class: `Wavelet_de_noising(signal, plot)`.<br/>
###### Arguments
`signal` - Time series or array without NAN values.<br/>
`plot`- Raw and filtered signal graph. `plot=True` for plotting, default is `plot=False`

#### Methods
1. `decompose(wavelet, level)` - main method for de-noising
    ###### Arguments
    `wavelet` - name of Wavelet function that is used for decomposition. Default `wavelet='db18'`<br/>
    `level` - level of decomposition. At each level, the signal is decomposed again into the same<br/>
    functions. Default `wavelet='db18'`<br/>

2. `get_wavelets` - additional method to get the names of all available wavelet functions<br/>
   (out of 127)

