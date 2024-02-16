from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def correlation_graph(pca,
                      x_y,
                      features) :
    """Affiche le graphe des correlations

    Positional arguments :
    -----------------------------------
    pca : sklearn.decomposition.PCA : notre objet PCA qui a été fit
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2
    features : list ou tuple : la liste des features (ie des dimensions) à représenter
    """

    # Extrait x et y
    x,y=x_y

    # Taille de l'image (en inches)
    fig, ax = plt.subplots(figsize=(10, 9))

    # Pour chaque composante :
    for i in range(0, pca.components_.shape[1]):

        # Les flèches
        ax.arrow(0,0,
                pca.components_[x, i],
                pca.components_[y, i],
                head_width=0.07,
                head_length=0.07,
                width=0.02, )

        # Les labels
        plt.text(pca.components_[x, i] + 0.05,
                pca.components_[y, i] + 0.05,
                features[i])

    # Affichage des lignes horizontales et verticales
    plt.plot([-1, 1], [0, 0], color='grey', ls='--')
    plt.plot([0, 0], [-1, 1], color='grey', ls='--')

    # Nom des axes, avec le pourcentage d'inertie expliqué
    plt.xlabel('F{} ({}%)'.format(x+1, round(100*pca.explained_variance_ratio_[x],1)))
    plt.ylabel('F{} ({}%)'.format(y+1, round(100*pca.explained_variance_ratio_[y],1)))

    # J'ai copié collé le code sans le lire
    plt.title("Cercle des corrélations (F{} et F{})".format(x+1, y+1))

    # Le cercle
    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(an), np.sin(an))  # Add a unit circle for scale

    # Axes et display
    plt.axis('equal')
    plt.show(block=False)
    
def display_factorial_planes(   X_projected,
                                x_y,
                                pca=None,
                                labels = None,
                                clusters=None,
                                alpha=1,
                                figsize=[10,8],
                                marker="." ):
    """
    Affiche la projection des individus

    Positional arguments :
    -------------------------------------
    X_projected : np.array, pd.DataFrame, list of list : la matrice des points projetés
    x_y : list ou tuple : le couple x,y des plans à afficher, exemple [0,1] pour F1, F2

    Optional arguments :
    -------------------------------------
    pca : sklearn.decomposition.PCA : un objet PCA qui a été fit, cela nous permettra d'afficher la variance de chaque composante, default = None
    labels : list ou tuple : les labels des individus à projeter, default = None
    clusters : list ou tuple : la liste des clusters auquel appartient chaque individu, default = None
    alpha : float in [0,1] : paramètre de transparence, 0=100% transparent, 1=0% transparent, default = 1
    figsize : list ou tuple : couple width, height qui définit la taille de la figure en inches, default = [10,8]
    marker : str : le type de marker utilisé pour représenter les individus, points croix etc etc, default = "."
    """

    # Transforme X_projected en np.array
    X_ = np.array(X_projected)

    # On définit la forme de la figure si elle n'a pas été donnée
    if not figsize:
        figsize = (7,6)

    # On gère les labels
    if  labels is None :
        labels = []
    try :
        len(labels)
    except Exception as e :
        raise e

    # On vérifie la variable axis
    if not len(x_y) ==2 :
        raise AttributeError("2 axes sont demandées")
    if max(x_y )>= X_.shape[1] :
        raise AttributeError("la variable axis n'est pas bonne")

    # on définit x et y
    x, y = x_y

    # Initialisation de la figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # On vérifie s'il y a des clusters ou non
    c = None if clusters is None else clusters

    # Les points
    # plt.scatter(   X_[:, x], X_[:, y], alpha=alpha,
    #                     c=c, cmap="Set1", marker=marker)
    sns.scatterplot(data=None, x=X_[:, x], y=X_[:, y], hue=c)

    # Si la variable pca a été fournie, on peut calculer le % de variance de chaque axe
    if pca :
        v1 = str(round(100*pca.explained_variance_ratio_[x]))  + " %"
        v2 = str(round(100*pca.explained_variance_ratio_[y]))  + " %"
    else :
        v1=v2= ''

    # Nom des axes, avec le pourcentage d'inertie expliqué
    ax.set_xlabel(f'F{x+1} {v1}')
    ax.set_ylabel(f'F{y+1} {v2}')

    # Valeur x max et y max
    x_max = np.abs(X_[:, x]).max() *1.1
    y_max = np.abs(X_[:, y]).max() *1.1

    # On borne x et y
    ax.set_xlim(left=-x_max, right=x_max)
    ax.set_ylim(bottom= -y_max, top=y_max)

    # Affichage des lignes horizontales et verticales
    plt.plot([-x_max, x_max], [0, 0], color='grey', alpha=0.8)
    plt.plot([0,0], [-y_max, y_max], color='grey', alpha=0.8)

    # Affichage des labels des points
    if len(labels) :
        # j'ai copié collé la fonction sans la lire
        for i,(_x,_y) in enumerate(X_[:,[x,y]]):
            plt.text(_x, _y+0.05, labels[i], fontsize='14', ha='center',va='center')

    # Titre et display
    plt.title(f"Projection des individus (sur F{x+1} et F{y+1})")
    plt.show()


def display_factorial_planes3D(X_projected, xyz, pca=None, labels=None, clusters=None, alpha=1, figsize=(10, 8), marker="."):
    """
    Display the projection of individuals in 3D.

    Args:
    X_projected (array-like): The matrix of projected points.
    xyz (list): The three dimensions (axes) to display, e.g., [0, 1, 2] for F1, F2, F3.
    pca (sklearn.decomposition.PCA): An initialized PCA object to display variance of each component. Default is None.
    labels (list): Labels of the individuals to project. Default is None.
    clusters (list): List of cluster assignments for each individual. Default is None.
    alpha (float): Transparency value in the range [0, 1]. Default is 1.
    figsize (tuple): Figure size (width, height) in inches. Default is (10, 8).
    marker (str): Marker type used to represent individuals. Default is ".".
    """

    # Convert X_projected to a NumPy array
    X_ = np.array(X_projected)

    # Check the dimensions
    if len(xyz) != 3:
        raise AttributeError("Three axes are required.")
    if max(xyz) >= X_.shape[1]:
        raise AttributeError("Invalid axis value.")

    # Initialize the figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Check if clusters are provided
    c = None if clusters is None else clusters

    # Plot the points
    ax.scatter(X_[:, xyz[0]], X_[:, xyz[1]], X_[:, xyz[2]], c=c, cmap="Set1", marker=marker, alpha=alpha)

    # Add labels to the points
    if labels is not None:
        for i, (x, y, z) in enumerate(X_[:, xyz]):
            ax.text(x, y, z + 0.05, labels[i], fontsize='14', ha='center', va='center')

    # Set the axis labels
    ax.set_xlabel(f'F{xyz[0]+1}')
    ax.set_ylabel(f'F{xyz[1]+1}')
    ax.set_zlabel(f'F{xyz[2]+1}')

    # Set the axis limits
    x_max = np.abs(X_[:, xyz[0]]).max() * 1.1
    y_max = np.abs(X_[:, xyz[1]]).max() * 1.1
    z_max = np.abs(X_[:, xyz[2]]).max() * 1.1
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_zlim(-z_max, z_max)

    def data_cleaning(data):

     print('======================\nDébut de la phase de cleaning\n======================')
    
     print('Gestion des valeurs inférieures à 0g et supérieures à 100g:')
     nan = 0
    
     for i in range(data.shape[0]):
         for j in range(2,len(data.columns)):
             if not np.isnan(data.iat[i,j]):
                 if ((data.iat[i,j]<0)|(data.iat[i,j]>100)):
                     data.iat[i,j] = np.nan
                     nan +=1
     print(nan,' valeurs aberrantes mises à NaN\n___________________________')
    
     print('Gestion des outliers pour la variable: energy_100g:')
     nan = data['energy_100g'].count()
     data['energy_100g']= data['energy_100g'].where(data['energy_100g']<3761)
    
     nan -= data['energy_100g'].count()
     print(nan,' valeurs aberrantes mises à NaN\n___________________________')
    
     print('Remplissage des NaNs via KNN Imputer (fitté sur le test)')
    

     scaler_1 =StandardScaler()
     train_scaled = scaler_1.fit_transform(data.drop(columns=['nutriscore_score']))
     scaled_data= scaler_1.transform(data.drop(columns=['nutriscore_score']))
    
     knn_imputer=KNNImputer(n_neighbors = 6)
     knn_imputer.fit(train_scaled)
     knn_data= knn_imputer.transform(scaled_data)
    
     filled_data = scaler_1.inverse_transform(knn_data)
    
     data_return = pd.DataFrame(filled_data, columns=data.columns[1:], index=data.index)
     data_return.insert(0,'nutriscore_score',data['nutriscore_score'].values)
     print('Les Nans du dataframe ont été imputés via KNNImputer\n___________________________')
    
     print('Suppression des lignes avec une somme des macronutriments supérieure à 100g:')
     suppr= data_return.shape[0]
     data_return['sum_100g']=(data_return['fat_100g']+data_return['carbohydrates_100g']+data_return['fiber_100g']
                              +data_return['proteins_100g']+data_return['salt_100g'])
     data_return= data_return.loc[data_return['sum_100g']<=100]
     data_return= data_return.drop(columns='sum_100g')
     suppr-=data_return.shape[0]
     print(suppr,' lignes supprimées\n___________________________')
    
     print('Suppression outliers multivariés avec KDTree:')
     suppr= data_return.shape[0]
    
     scaler_2 = StandardScaler()
     train_scaled2 = scaler_2.fit_transform(data)
     scaled_data = scaler_2.transform(data_return)
    
     scaled_tree = spatial.KDTree(train_scaled2)
     neighbours_scaled = scaled_tree.query(scaled_data,k=6)
     dist_scaled = pd.DataFrame(neighbours_scaled[0])
     dist_scaled = dist_scaled.drop(columns=0)
     dist_scaled['mean']=dist_scaled.mean(axis=1)
     data_return['mean']=dist_scaled['mean'].values
     data_return= data_return.loc[data_return['mean']<0.84]
     suppr-= data_return.shape[0]
     print(suppr,' lignes supprimées\n___________________________')
     data_return = data_return.drop(columns=['mean'])
    
    
     return data_return
    