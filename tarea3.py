
"""
Created on Thu Jun 25 23:07:35 2020

@author: Pamela Barquero B60925
"""
import numpy as np
from scipy import stats
import matplotlib.pyplot  as plt
import pandas as pd
import csv
from numpy import linspace
from mpl_toolkits.mplot3d import Axes3D
#Extracción de los datos modificados del xy.csv mediante pandas
datos= pd.read_csv('xy.csv')
df=pd.DataFrame(datos)
print(df)
# A partir de los datos, encontrar la mejor curva de ajuste (modelo 
#probabilístico) para las funciones de densidad marginales de X y Y.
fx= np.sum(df,axis=1)
fy=np.sum(df,axis=0)
print(fx)
print(fy)
x=linspace(5,15,11)
y=linspace(5,25,21)
print(x)
print(y)
ajustex= plt.plot(x,fx)
plt.title('Curva de ajuste de X')
plt.savefig('Curva de ajuste de X')
plt.cla()
ajustey=plt.plot(y,fy)
plt.title('Curva de ajuste de Y')
plt.savefig('Curva de ajuste de Y')
plt.cla()
# Al graficar las pmf marginales se observa que ambas tienen forma gaussiana
#Modelo probabilístico gaussiano para X
mu, sigma = 5, 3
gauss = stats.norm(mu, sigma)
x = np.linspace(gauss.ppf(0.01),
                gauss.ppf(0.99), 100)
fp = gauss.pdf(x) # Función de Probabilidad
plt.plot(fx)
plt.plot(x, fp)

plt.title('Gaussiana')
plt.savefig('Comparación de las curvas X')
plt.cla()
#Modelo probabilístico gaussiano para Y
mu1, sigma1 = 10, 5
gauss = stats.norm(mu1, sigma1)
x = np.linspace(gauss.ppf(0.01),
                gauss.ppf(0.99), 100)
fp = gauss.pdf(x)
plt.plot(fy)
plt.plot(x, fp)
plt.title('Gaussiana')
plt.savefig('Comparación de las curvas Y')
plt.cla()

# Asumir independencia de X y Y. Analíticamente, ¿cuál es entonces la 
#expresión de la función de densidad conjunta que modela los datos?

#la expresión encontrada se muestra en una imagen adjunta en el GitHub


#Hallar los valores de correlación, covarianza y coeficiente de correlación
# (Pearson) para los datos y explicar su significado
#Obtensión de resultados a partir del xy.csv
#Creación de vectores de X y Y con valores requeridos
y = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
x = [5,6,8,9,10,11,12,13,14,15]
df = []
with open('xy.csv', newline='') as doc:
	datos = csv.reader(doc)
	next(datos, None) 
	for filas in datos:
		df.append(filas)
#Para obtener la correlación:
correlacion = 0;
for j in range(20):
      for i in range(10):
          if j == 0:
                  j = j + 1;
          else:
                  prom = float(df[i][j]);
                  vecy = y[j];
                  vecx = x[i];
                  correlacion += prom*vecx*vecy;
                  


print ("Correlación:",correlacion)
#Para obtener la covarianza:
sumvecx = 0;
sumvecy = 0;
for j in range(20):
  sumvecy += y[j];
for i in range(10):
  sumvecx +=x[i];

mediax = sumvecx/11;
mediay = sumvecy/21;

print('Media x:',mediax)
print('Media y:',mediay)

covarianza = 0;
prom = 0;
for j in range(20):
      for i in range(10):
          if j == 0:
                  j = j + 1;
          else:
                  prom = float(df[i][j]);
                  yvector = y[j] - mediax;
                  xvector = x[i] - mediay;
                  covarianza += prom*yvector*xvector;
                  
print("Covarianza:", covarianza)
print("Se observa que hay una asociación lineal negativa entre X y Y")
#Para obtener el coeficiente de correlación de pearson:
varianzax=yvector/11;
varianzay=xvector/21;

pearson = covarianza/varianzax*varianzay

print("Coeficiente de correlación de Pearson:",pearson)

print("Se observa que hay un porcentaje de asociación lineal negativa entre X y Y")

# Graficar las funciones de densidad marginales (2D), la función de 
#densidad conjunta (3D).
#Extracción de los datos ordenados del xyp.csv mediante pandas
xyp=pd.read_csv("xyp.csv")
dfp= xyp.to_numpy()
xyp = np.array(dfp).astype("float")


X=xyp[:,0]
Y=xyp[:,1]
Z=xyp[:,2]

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(X,Y,Z)
plt.show()
plt.savefig('Densidad conjunta 3D')
