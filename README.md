# Hybrid model with TSK, VAR time series and Wavelet de-noising
 We present a model for forecasting financial time series. It was created <br/>
 to forecast stock returns, but can be widely used in other areas. The model<br/>
 is based on the fuzzy system of Takagi, Sugeno and Kang, which provides<br/>
 a polynomial dependence with soft switching between equations. <br/>
 Vector autoregression (VAR) is used to obtain predictions of regressors.<br/>
 Wavelet decomposition is used as preprocessing and clears time series from noise.<br/>
 We also use FCM fuzzy clustering to split the data into fuzzy sets and make it ready for<br/>
 the TSK model use.

### model diagram:
1. [Data de-noising with Wavelet decomposition](./Wavelet_de_noising.py)
2. [VAR forecast for factors](./VAR_estimation.R)
3. [Fuzzy clustering with FCM algorithm](./FCM_fuzzy_clustering.py)
4. [Takagi-Sugeno_Kang fuzzy system forecast](./Takagi_Sugeno_Kang.py)
