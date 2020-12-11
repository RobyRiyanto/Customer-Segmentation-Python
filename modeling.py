from kmodes.kmodes import KModes  
from kmodes.kprototypes import KPrototypes  
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

class Model_segmen:
    def __init__(self, df_model):
        self.df_model = df_model

    def find_optimalCluster(self): # Mencari Jumlah Cluster yang Optimal
        # Melakukan Iterasi untuk Mendapatkan nilai Cost  
        cost = {}  
        for k in range(2,10):  
            kproto = KPrototypes(n_clusters = k, random_state = 75)  
            kproto.fit_predict(self.df_model, categorical = [0,1,2])  
            cost[k]= kproto.cost_
        
        # Memvisualisasikan Elbow Plot  
        sns.pointplot(x = list(cost.keys()), y = list(cost.values()))  
        plt.show()

    def making_model(self):
        kproto = KPrototypes ( n_clusters = 5, random_state = 75)  
        kproto = kproto.fit(self.df_model, categorical=[0,1,2])  
        
        #Save Model  
        pickle.dump(kproto, open('cluster.pkl', 'wb'))

        self.kproto = kproto

    def use_model(self):
        df = pd.read_csv("https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/customer_segments.txt", sep="\t") 
        # Menentukan segmen tiap pelanggan    
        clusters =  self.kproto.predict(self.df_model, categorical=[0,1,2])    
        print('segmen pelanggan: {}\n'.format(clusters))    
            
        # Menggabungkan data awal dan segmen pelanggan    
        df_final = df.copy()    
        df_final['cluster'] = clusters
        print(df_final.head())

        self.df_final = df_final

    def Showing_EachCustomerCluster(self):
        # Menampilkan data pelanggan berdasarkan cluster nya  
        for i in range (0,5):  
            print('\nPelanggan cluster: {}\n'.format(i))
            print(self.df_final[self.df_final['cluster'] == i])

    def VisualizationClusteringResults_BoxPlot(self):
        # Data Numerical
        kolom_numerik = ['Umur','NilaiBelanjaSetahun']  
        
        for i in kolom_numerik:  
            plt.figure(figsize=(6,4))  
            ax = sns.boxplot(x = 'cluster',y = i, data = self.df_final)  
            plt. title('\nBox Plot {}\n'.format(i), fontsize=12)  
            plt.show()

    def VisualizationClusteringResults_CountPlot(self):
        # Data Kategorikal  
        kolom_categorical = ['Jenis Kelamin','Profesi','Tipe Residen']  
        
        for i in kolom_categorical:  
            plt.figure(figsize=(6,4))  
            ax = sns.countplot(data = self.df_final, x = 'cluster', hue = i )  
            plt.title('\nCount Plot {}\n'.format(i), fontsize=12)  
            ax. legend (loc="upper center")  
            for p in ax.patches:  
                ax.annotate(format(p.get_height(), '.0f'),  
                            (p.get_x() + p.get_width() / 2., p.get_height()),  
                            ha = 'center',  
                            va = 'center',  
                            xytext = (0, 10),  
                            textcoords = 'offset points')  
            
            sns.despine(right=True, top = True, left = True)  
            ax.axes.yaxis.set_visible(False)  
            plt.show()

    def NamingCluster(self):
        # Mapping nama kolom  
        self.df_final['segmen'] = self.df_final['cluster'].map({  
            0: 'Diamond Young Member',  
            1: 'Diamond Senior Member',  
            2: 'Silver Member',  
            3: 'Gold Young Member',  
            4: 'Gold Senior Member'  
        })

        print(self.df_final.info())
        print(self.df_final.head())

        # Save to CSV
        self.df_final.to_csv (r'data\df-customer-segmentation-final.csv', index = False)
        print('\nDataframe Sudah Tersimpan!')

df_model = pd.read_csv('https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/df-customer-segmentation.csv')

app = Model_segmen(df_model)
# app.find_optimalCluster()
app.making_model()
app.use_model()
app.Showing_EachCustomerCluster()
# app.VisualizationClusteringResults_BoxPlot()
# app.VisualizationClusteringResults_CountPlot()
app.NamingCluster()