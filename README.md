# Hybrid fuzzy model for Stock returns forecasting
 We present a model for forecasting financial time series. It was created <br/>
 to forecast stock returns, but can be widely used in other areas. The model<br/>
 is based on the fuzzy system of Takagi, Sugeno and Kang, which provides<br/>
 a polynomial dependence with soft switching between equations. <br/>
 Vector autoregression (VAR) is used to obtain predictions of regressors.<br/>
 Wavelet decomposition is used as preprocessing and clears time series from noise.<br/>
 We also use FCM fuzzy clustering to split the data into fuzzy sets and make it ready for<br/>
 the TSK fuzzy model use.

### model structure:
1. [Data de-noising with Wavelet decomposition](./Wavelet_de_noising.py)
2. [VAR forecast for factors](./VAR_estimation.R)
3. [Fuzzy clustering with FCM algorithm](./FCM_fuzzy_clustering.py)
4. [Takagi-Sugeno_Kang fuzzy system](./Takagi_Sugeno_Kang.py)

## Wavelet decomposition
Financial time series have noise that interferes with the identification of trends.<br/>
Also, noise, outliers and heteroscedasticity make the time series weakly stationary.<br/>
Therefore, we use wavelet decomposition for de-noising. The main idea of this approach is<br/>
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

## FCM fuzzy clustering
In order to apply fuzzy methods, it is necessary to determine the membership function of elements<br/>
to fuzzy sets. In our case, we use the fuzzy clustering technique to determine the membership matrix.<br/>
The idea is to use unsupervised learning to split the dataset into fuzzy sets. As a result, each<br/>
observation belongs to each of the clusters (sets) with some degree. We use the FCM algorithm for this<br/>
task. In fact, it is a modification of the K-means algorithm. In K-means, the element belongs to the<br/>
cluster with the nearest centroid, and the centroids themselves are recalculated as the average of the<br/>
clusters. In FCM, elements belong to all clusters at the same time. Therefore, we calculate the<br/>
membership matrix, and the centroids are updated as the weighted average of the clusters according to<br/>
this membership matrix.

### Code documentation for FCM
#### Class
The FCM algorithm is written as a class: `FCM(K, max_iters)`.<br/>
###### Arguments
`K` - Number of clusters returning as output. Default `K=3`<br/>
`max_iters`- Maximum number of iterations. Default `max_iters=100'`<br/>

#### Methods
1. `predict(X)`
    ###### Arguments
    `X` - Dataset or variable for clustering. Better to use nupy array. <br/>
    Method returns membership matrix as output.<br/>

## Takagi Sugeno Kang fuzzy model
This model uses the consequent equations, each with its own functional dependence. The equations<br/>
relate to their clusters. The weight of each equation in the final forecast depends on the degree<br/>
to which the observation belongs to these clusters. Gradient descent optimization involves two stages.<br/>
At the first stage, models with all possible combinations of clusters and equations are evaluated.<br/>
At the second stage, the best combination is selected. Thus, we obtain a polynomial dependence that <br/>
dynamically changes depending on the membership matrix of observations to clusters.

### Code documentation for TSK model
#### Class
The TSK is written as a class: `TakagiSugeno(cluster_n, lr, n_iters)`.<br/>
###### Arguments
`cluster_n` - Number of clusters in membership matrix and also number of consequent equations.<br/>
              Default `cluster_n=2`<br/>
`lr`- Learning rate. Default `lr=0.01'`<br/>   
`n_iters`- Maximum number of iterations. Default `n_iters=1500'`<br/> 
