# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 11:06:28 2020

@author: mofoko
"""



#importamos los paquetes necesarios
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split

#definimos la ruta de los data sets consultadas el 22.09.2020
Path = 'C:/Users/mofoko/Desktop/Python/Projecto Final/'


Link_DB_IHFC = Path + 'HeatFlowIHFC_Panama.txt'
Link_Pollack = Path + 'Oceanic_Heat_Flow_(Pollack_et_al_1993).csv'

#cargamos los data set como DataFrame

df_DB_IHFC = pd.read_csv(Link_DB_IHFC, sep = ';')
df_Pollack = pd.read_csv(Link_Pollack)

#Exploración de la informacion
df_DB_IHFC.info()
df_Pollack.info()


df_DB_IHFC.head()
df_Pollack.head()

#df_DB_IHFC[df_DB_IHFC['Site_Name'] == df_Pollack['label']].count()
#Repetidos = pd.Series(np.where(df_DB_IHFC['Site_Name'] == df_Pollack['label'], 'True', 'False'))

#Manipulación de datos
df_DB_IHFC = pd.concat([df_DB_IHFC['Site_Name'], 
                        df_DB_IHFC['Longitude'], 
                        df_DB_IHFC['Latitude'],
                        df_DB_IHFC['Elevation'], 
                        df_DB_IHFC['Heat_Flow']], axis= 1)
df_DB_IHFC 

df_Pollack = pd.concat([df_Pollack['Site Name'], 
                        df_Pollack['Long'], 
                        df_Pollack['Lat'],
                        df_Pollack['Elevation'], 
                        df_Pollack['Heat Flow']], axis= 1)
df_Pollack

#Cambio de nombre de las columnas
df_DB_IHFC.rename(columns= {'Site_Name':'Codigo',
                            'Longitude':'lon',
                            'Latitude':'lat', 
                            'Elevation':'Profundidad', 
                            'Heat_Flow':'Flujo_calor'}, inplace= True)

df_Pollack.rename(columns= {'Site Name':'Codigo',
                            'Long':'lon',
                            'Lat':'lat',
                            'Elevation':'Profundidad', 
                            'Heat Flow':'Flujo_calor'}, inplace= True)

#Se buscan los valores nulos

df_Pollack.isna().sum()
df_DB_IHFC.isna().sum()

#Eliminamos los valores nulos

df_Pollack = df_Pollack.dropna()

#Eliminamos los valores iguales a cero
df_Pollack = df_Pollack[(df_Pollack != 0).all(1)]
df_DB_IHFC = df_DB_IHFC[(df_DB_IHFC != 0).all(1)]

#Fusionamos los DataFrame

df_Flujo_calor = pd.concat([df_Pollack, df_DB_IHFC], ignore_index= True)

df_Flujo_calor.info()

#Buscamos datos duplicados
df_Flujo_calor.duplicated(subset=['Codigo']).sum()

#La muestra RIS-41 esta dos veces
df_Flujo_calor[df_Flujo_calor['Codigo'] == 'RIS-41']

#promediamos las muestras con codigos iguales

df_Flujo_calor = df_Flujo_calor.groupby('Codigo').mean().reset_index()
df_Flujo_calor.duplicated(subset=['Codigo']).sum()

#se observa que los primeros 4 valores tienen valores inconsistentes de profundidad por lo que se procede a eliminarlos
df_Flujo_calor = df_Flujo_calor.drop([0, 1, 2, 3],axis=0)
df_Flujo_calor.reset_index(drop= True, inplace= True)

df_Flujo_calor.info()
df_Flujo_calor.head()

#medidas de tendencia central

df_Flujo_calor.describe()
df_Flujo_calor[df_Flujo_calor['Flujo_calor'] < 0]

#scater matrix
sns.pairplot(df_Flujo_calor);

#se observa que hay 2 valores inconsistentes de flujo de calor (filas 49 y 51)
#se porcede a eliminar estas muestras
df_Flujo_calor = df_Flujo_calor.drop([49, 51], axis=0)

df_Flujo_calor[df_Flujo_calor['Flujo_calor'] < 0].count()

#Medidas de tendencia central
df_Flujo_calor.describe()

#Graficas geográficas
fig = px.density_mapbox(df_Flujo_calor, lat='lat', lon='lon', z='Flujo_calor', radius=10,
                        center=dict(lat=2.5, lon=-86), zoom=4.2,
                        mapbox_style="carto-darkmatter", title='Flujo de Calor en la Cuenca Panamá')
fig.show()
fig.write_html(Path + 'Flujo_Cal_Cuenca')

#con Batimetría
fig = px.density_mapbox(df_Flujo_calor, lat='lat', lon='lon', z='Flujo_calor', radius=20,
                        center=dict(lat=2.5, lon=-86), zoom=5)
fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": 'traces',
            "sourcetype": "raster",
            "sourceattribution": "United States Geological Survey",
            "source": [
                "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
            ]
        }
      ])
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
fig.write_html(Path + 'Flujo_Cal_Cuenca_Bati')

#Dispersión 3D
fig = px.scatter_3d(df_Flujo_calor, x='lon', y='lat', z='Profundidad',
                    color='Flujo_calor', size='Flujo_calor', size_max=100, title='Distribución espacial de los datos de Flujo de Calor')
fig.show()
fig.write_html(Path + 'Dispersión 3D')

#Histogramas
fig = px.histogram(df_Flujo_calor, x="Flujo_calor", marginal="violin", title='Histograma Flujo de Calor')
fig.show()
fig.write_html(Path + 'Hist_Calor')

fig = px.histogram(df_Flujo_calor, x="Profundidad", marginal="violin", title='Profundidad')
fig.show()
fig.write_html(Path + 'Hist_Profundidad')

#Box

fig = px.box(df_Flujo_calor,
             x = 'Flujo_calor',
             orientation = 'h',
             notched = True, 
             points = 'all', 
             title = 'Distribución de los Datos de Flujo de Calor')

fig.show()
fig.write_html(Path + 'Box_Calor')

fig = px.box(df_Flujo_calor,
             x = 'Profundidad',
             orientation = 'h',
             notched = True, 
             points = 'all', 
             title = 'Distribución de los Datos de Profundidad')

fig.show()
fig.write_html(Path + 'Box_Profundidad')

#scatter Calor vs. Profundidad

fig = px.scatter(df_Flujo_calor, x="Flujo_calor", y="Profundidad", size="Flujo_calor", color="Flujo_calor", trendline='ols', marginal_x='histogram', marginal_y='histogram',)
fig.show()
fig.write_html(Path + 'Dispersión_CalvsProf')

#Prueba de Levene
statistic, pvalue= stats.levene(df_Flujo_calor['Flujo_calor'], df_Flujo_calor['Profundidad'])
pvalue= round(pvalue)
print(f'El p-valor:{pvalue} es mayor a al nivel de significancia por lo que NO RECHAZAMOS H0') if pvalue >= 0.05 else print(f'El p-valor:{pvalue} es menor a al nivel de significancia por lo que RECHAZAMOS H0')

#Pruebas de Normalidad

#Shapiro-Wilk
statistic, pvalue= stats.shapiro(df_Flujo_calor['Flujo_calor'])
pvalue= round(pvalue)
print(f'El p-valor:{pvalue} es mayor a al nivel de significancia por lo que NO RECHAZAMOS H0') if pvalue >= 0.05 else print(f'El p-valor:{pvalue} es menor a al nivel de significancia por lo que RECHAZAMOS H0')

#qqplot
stats.probplot(df_Flujo_calor['Flujo_calor'], plot = plt, dist= 'norm')
plt.title('Gráfico Q-Q del Flujo de calor') 
plt.show()

#Correlación para el análisis de correlación entre las variables se usa el método de Spearman, dado que las variables no presentan distribución normal 

df_corr= df_Flujo_calor.corr()
df_corr

fig = px.imshow(df_corr)
fig.write_html(Path + 'Matriz de Corr')

#Discretizar por edad
l1= [139.5, 109.1, 81.9, 62.3, 61.7, 65.1, 61.5, 56.3, 53.0, 51.3]
l2= [109.1, 81.9, 62.3, 61.7, 65.1, 61.5, 56.3, 53.0, 51.3]
max = [df_Flujo_calor['Flujo_calor'].max()]
min = [df_Flujo_calor['Flujo_calor'].min()]
l3= list(map(lambda item1, item2: (item1 + item2)/2, l1, l2))

valor_medio= max + l3 + min


Periodo_geo= ['Quaternary', 'Pliocene', 'Miocene', 'Oligocene', 
              'Eocene', 'Paleocene', 'Late Cretaceous', 'Middle Cretaceous', 
              'Early Cretaceous', 'Late Jurassic']



df_Flujo_calor['Etapa_termica'] = df_Flujo_calor.Flujo_calor.map( lambda x: Periodo_geo[0] if valor_medio[0] >= x >= valor_medio[1] else Periodo_geo[1] if valor_medio[1] >= x >= valor_medio[2] else Periodo_geo[2] if valor_medio[2] >= x >= valor_medio[3] else Periodo_geo[3] if valor_medio[3] >= x >= valor_medio[4] else Periodo_geo[4] if valor_medio[4] >= x >= valor_medio[5] else Periodo_geo[5] if valor_medio[5] >= x >= valor_medio[6] else Periodo_geo[6] if valor_medio[6] >= x >= valor_medio[7] else Periodo_geo[7] if valor_medio[7] >= x >= valor_medio[8] else Periodo_geo[8] if valor_medio[8] >= x >= valor_medio[9] else Periodo_geo[9] if valor_medio[9] >= x >= valor_medio[10] else False) 
df_Flujo_calor.head()


fig = px.bar(df_Flujo_calor, x='Etapa_termica', y='Flujo_calor', color='Flujo_calor', labels='Flujo_calor', log_y=True)
fig.update_layout(barmode='stack', xaxis={'categoryorder':'array', 'categoryarray':Periodo_geo}) #'category descending'
fig.show()

def pvalor(x):
    cols= ['lon', 'lat', 'Profundidad', 'Flujo_calor']
    c= []
    pv= []
    for i in cols:
        coef, p = stats.spearmanr(df_Flujo_calor[x], df_Flujo_calor[i])
        c.append(coef)
        pv.append(p)
        c = [round(num0, 1) for num0 in c]
        pv= [round(num1, 1) for num1 in pv]
        #pv = [round(num, 6) for num in pv]
        
    return print(f'p-valores: {pv} \ncoeficientes de Correlación: {c} \nPara: 1. longitud, 2. latitud, 3. Profundidad y 4. Flujo de calor respectivamente')

pvalor('lon')
pvalor('lat')
pvalor('Profundidad')
pvalor('Flujo_calor')



#Convertimos a variable numperica la varible Etapa_térmica
df_Flujo_calor['No_etapa']= df_Flujo_calor.Etapa_termica.map( lambda x: 1 if x == Periodo_geo[0] else 2 if x == Periodo_geo[1] else 3 if x == Periodo_geo[2] else 4 if x == Periodo_geo[3] else 5 if x == Periodo_geo[4] else 6 if x == Periodo_geo[5] else 7 if x == Periodo_geo[6] else 8 if x == Periodo_geo[7] else 9 if x == Periodo_geo[8] else 10 if x == Periodo_geo[9] else False) 
df_Flujo_calor.head()

#creamos la matriz scater
varnumericas = df_Flujo_calor.columns[1:5].tolist()
varnumericas

from matplotlib import cm

X = df_Flujo_calor[varnumericas]
y = df_Flujo_calor['No_etapa']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

cmap = ListedColormap(['#FEF2E0', '#FFFF99', '#FFFF00','#FDC07A', '#FDB46C', '#FDA75F', '#A6D84A', '#7FC64E', '#8CCD57', '#B3E3EE'])
scatter = pd.plotting.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(10,10), cmap=cmap, alpha=1)
patch0 = mpatches.Patch(color='#FEF2E0', label='Quaternary')
patch1 = mpatches.Patch(color='#FFFF99', label='Pliocene')
patch2 = mpatches.Patch(color='#FFFF00', label='Miocene')
patch3 = mpatches.Patch(color='#FDC07A', label='Oligocene')
patch4 = mpatches.Patch(color='#FDB46C', label='Eocene')
patch5 = mpatches.Patch(color='#FDA75F', label='Paleocene')
patch6 = mpatches.Patch(color='#A6D84A', label='Late Cretaceous')
patch7 = mpatches.Patch(color='#7FC64E', label='Middle Cretaceous')
patch8 = mpatches.Patch(color='#8CCD57', label='Early Cretaceous')
patch9 = mpatches.Patch(color='#B3E3EE', label='Late Jurassic')
plt.legend(handles=[patch0, patch1, patch2, patch3, patch4, patch5, patch6, patch7, patch8, patch9], bbox_to_anchor=(1.05, 1))

# plotting a 3D scatter plot


cmap = ['#FEF2E0', '#FFFF99', '#FFFF00','#FDC07A', '#FDB46C', '#FDA75F', '#A6D84A', '#7FC64E', '#8CCD57', '#B3E3EE']
fig = go.Figure(data=[go.Scatter3d(
    x=X_train['lon'],
    y=X_train['lat'],
    z=X_train['Profundidad'],
    mode='markers',
    marker=dict(
        size=7,
        color=y_train,               # set color to an array/list of desired values
        colorscale=cmap,    # choose a colorscale
        opacity=0.8
    )
)])
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_html(Path + 'scatter3d_Etapas')

#Create train-test split
# colcoamos las variables en el x y en y el clasificador
X = df_Flujo_calor[varnumericas]
y = df_Flujo_calor['No_etapa']

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Creamos un objeto clasificador
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

#Entrenamos el clasificador
knn.fit(X_train, y_train)

#Calculamos la exactitud del clasificador
knn.score(X_test, y_test) 

#Probamos el clasificador con una muestra que no este en el Dataset
# relacionamos la etapa termica con un numero
nombre_etapa_termica = dict(zip(df_Flujo_calor.No_etapa.unique(), df_Flujo_calor.Etapa_termica.unique()))   
nombre_etapa_termica

# Ejemplo sea una muestra con lon= promedio, lat= prmedio, Profundidad= promedio, Flujo_calor= promerio
Etapa_prediccion = knn.predict([[df_Flujo_calor['lon'].mean(), df_Flujo_calor['lat'].mean(), df_Flujo_calor['Profundidad'].mean(), df_Flujo_calor['Flujo_calor'].mean()]])
nombre_etapa_termica[Etapa_prediccion[0]]

#Graficamos los limites de desicion de los k-NN classifier

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier
X = df_Flujo_calor[['lon', 'lat', 'Profundidad', 'Flujo_calor']]
y = df_Flujo_calor['No_etapa']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

def plot_EtapasCalor_knn(X, y, n_neighbors, weights):
    X_mat = X[['lon', 'lat']].values
    y_mat = y.values
# Create color maps
    cmap_light = ListedColormap(['#FDF2EC', '#FFFFBF', '#FEFF73','#FEE6AA', '#FDCCA1', '#FDBE6F', '#F2FA8B', '#B3DD53', '#A6D875', '#DAF1F7'])
    cmap_bold  = ListedColormap(['#FEF2E0', '#FFFF99', '#FFFF00','#FDC07A', '#FDB46C', '#FDA75F', '#A6D84A', '#7FC64E', '#8CCD57', '#B3E3EE'])
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    #clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)
# Plot the decision boundary by assigning a color in the color map
    # to each mesh point.

    mesh_step_size = .01  # step size in the mesh
    plot_symbol_size = 50

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
# Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor = 'black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    patch0 = mpatches.Patch(color='#FEF2E0', label='Quaternary')
    patch1 = mpatches.Patch(color='#FFFF99', label='Pliocene')
    patch2 = mpatches.Patch(color='#FFFF00', label='Miocene')
    patch3 = mpatches.Patch(color='#FDC07A', label='Oligocene')
    patch4 = mpatches.Patch(color='#FDB46C', label='Eocene')
    patch5 = mpatches.Patch(color='#FDA75F', label='Paleocene')
    patch6 = mpatches.Patch(color='#A6D84A', label='Late Cretaceous')
    patch7 = mpatches.Patch(color='#7FC64E', label='Middle Cretaceous')
    patch8 = mpatches.Patch(color='#8CCD57', label='Early Cretaceous')
    patch9 = mpatches.Patch(color='#B3E3EE', label='Late Jurassic')
    plt.legend(handles=[patch0, patch1, patch2, patch3, patch4, patch5, patch6, patch7, patch8, patch9], bbox_to_anchor=(1.05, 1))
plt.xlabel('Longitud')
plt.ylabel('Latitud')
##plt.title("4-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))    
#plt.show()
#plot_EtapasCalor_knn(X_train, y_train, 5, 'uniform')

#Corremos el modelo para diferentes numeros de Vecinos
plot_EtapasCalor_knn(X_train, y_train, 5, 'uniform')   # Para 5 vecinos más cercanos
plot_EtapasCalor_knn(X_train, y_train, 2, 'uniform')   # Para 2 vecinos más cercanos
plot_EtapasCalor_knn(X_train, y_train, 19, 'uniform')   # Para 19 vecinos más cercanos
plot_EtapasCalor_knn(X_train, y_train, 14, 'uniform')   # Para 14 vecinos más cercanos
plot_EtapasCalor_knn(X_train, y_train, 10, 'uniform')   # Para 10 vecinos más cercanos


#Que tan sensitivo es nuestro KNN al valor K
k_range = range(1,20)
scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0,5,10,15,20]);

#que tan sensitivo la exacttud de es nuestra clasificación k-NN a la proporción entrenamiento/test split?
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

knn = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn.fit(X_train, y_train)
        scores.append(knn.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');