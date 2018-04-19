import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt
from random import gauss from matplotlib.pylab
import rcParams rcParams['figure.figsize'] = 12, 10 
 
 #initialisation  
 N = 500 
 T = 1. sh = np.sqrt(T/N) 
 y = np.zeros(N) 
 bruitY = [gauss(0.,np.sqrt(T)) for i in range(N)] 
 for i in range(1,N):     y[i] = y[i-1] + sh*bruitY[i-1] 
 x = np.arange(0, 1, 1/N) np.random.seed(10) data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y']) plt.plot(data['x'],data['y'],'.') 
 for i in range(2,16):       colname = 'x_%d'%i           data[colname] = data['x']**i 
 
 
#Importons le modèle de régression linéaire depuis scikit-learn.
from sklearn.linear_model import LinearRegression
def linear_regression(data, power, models_to_plot):
#On initialise les predicteurs:
predictors=['x'] 
if power>=2: 
predictors.extend(['x_%d'%i for i in range(2,power+1)])
#On "fit" le modèle 
linreg = LinearRegression(normalize=True)
linreg.fit(data[predictors],data['y'])
y_pred = linreg.predict(data[predictors]) 

  if power in models_to_plot:
  plt.subplot(models_to_plot[power]) 
  plt.tight_layout() 
  plt.plot(data['x'],y_pred) 
  plt.plot(data['x'],data['y'],'.')
  plt.title('Plot for power: %d'%power) 
  #on renvoie le résultat dans un format adapté 
  rss = sum((y_pred-data['y'])**2)
  ret = [rss] 
  ret.extend([linreg.intercept_])
  ret.extend(linreg.coef_)
  return ret 
 #On initialise le dataframe pour stocker les résultats:
 col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
 ind = ['model_pow_%d'%i for i in range(1,16)]
 coef_matrix_simple = pd.DataFrame(index=ind, columns=col) 
 
 
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236} 
 #On affiche les graphiques for i in range(1,16):
 coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data, power=i, models_to_plot=models_to_plot)
 #On affiche les coefficients
 pd.options.display.float_format = '{:,.2g}'.format
 print(coef_matrix_simple) 
 from sklearn.linear_model import Ridge 
 def ridge_regression(data, predictors, rho, models_to_plot={}): 
 #On "fit" le modèle
 ridgereg = Ridge(alpha=rho,normalize=True)
 ridgereg.fit(data[predictors],data['y'])
 y_pred = ridgereg.predict(data[predictors])      
     if rho in models_to_plot:
     plt.subplot(models_to_plot[rho]) 
     plt.tight_layout()     
     plt.plot(data['x'],y_pred)  
     plt.plot(data['x'],data['y'],'.')    
     plt.title('Plot for rho: %.3g'%rho)  
     #on renvoie le résultat dans un format adapté  
     rss = sum((y_pred-data['y'])**2)
     ret = [rss] 
     ret.extend([ridgereg.intercept_]) 
     ret.extend(ridgereg.coef_)  
     return ret 
 
predictors=['x'] 

predictors.extend(['x_%d'%i for i in range(2,16)]) 
 #Les différentes valeurs de rho à tester
 rho_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20] 
 #On initialise le dataframe.
 col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
 ind = ['rho_%.2g'%rho_ridge[i] for i in range(0,10)]
 coef_matrix_ridge = pd.DataFrame(index=ind, columns=col) 
 models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 5:236}
 for i in range(10):
 coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, rho_ridge[i], models_to_plot)   




