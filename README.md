# China-GDP-analysis

This project is to apply ARMA model to China GDP annual data from 1992 to 2015.
First, I do stationary test on level data and log return using Augmented Dickey-Fuller Test. 
Then, I  uses statsmodels package to fit ARMA model to select the best order.
It also captures GARCH effect and select best order for GARCH.
The model is tested using test group and draws forecast line compared with true data line.
