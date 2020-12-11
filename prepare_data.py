import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class Cust_segment:
    def __init__(self):
        self.data = data

    def read_data(self):
        # import dataset  
        df = pd.read_csv(data, sep="\t")  
        
        # menampilkan data  
        print(df.head())

        # Menampilkan informasi data  
        print(df.info())

        return df

    def eksplorasi_dataNumerik(self, df):
        sns.set(style='white')
        plt.clf()
        
        # Fungsi untuk membuat plot  
        def observasi_num(features):  
            fig, axs = plt.subplots(2, 2, figsize=(10, 9))
            for i, kol in enumerate(features):
                sns.boxplot(df[kol], ax = axs[i][0])
                sns.distplot(df[kol], ax = axs[i][1])
                axs[i][0].set_title('mean = %.2f\n median = %.2f\n std = %.2f'%(df[kol].mean(), df[kol].median(), df[kol].std()))
            plt.setp(axs)
            plt.tight_layout()
            plt.show()  

        # Memanggil fungsi untuk membuat Plot untuk data numerik  
        kolom_numerik = ['Umur','NilaiBelanjaSetahun'] 
        observasi_num(kolom_numerik)

    def eksplorasi_dataKategorikal(self, df):
        sns.set(style='white')
        plt.clf()
        
        # Menyiapkan kolom kategorikal  
        kolom_kategorikal = ['Jenis Kelamin','Profesi','Tipe Residen']  

        # Membuat canvas
        fig, axs = plt.subplots(3,1,figsize=(7,10)) 

        # Membuat plot untuk setiap kolom kategorikal  
        for i, kol in enumerate(kolom_kategorikal):  
            # Membuat Plot
            sns.countplot(df[kol], order = df[kol].value_counts().index, ax = axs[i])  
            axs[i].set_title('\nCount Plot %s\n'%(kol), fontsize=15)  
            
            # Memberikan anotasi  
            for p in axs[i].patches:  
                axs[i].annotate(format(p.get_height(), '.0f'),  
                                (p.get_x() + p.get_width() / 2., p.get_height()),  
                                ha = 'center',  
                                va = 'center',  
                                xytext = (0, 10),  
                                textcoords = 'offset points') 
                
            # Setting Plot  
            sns.despine(right=True,top = True, left = True)  
            axs[i].axes.yaxis.set_visible(False) 
            plt.setp(axs[i])
        #     plt.setp(ax)
            plt.tight_layout()

        # Tampilkan plot
        plt.show()
    
    def preparation_data(self, df):
        # Standarisasi Kolom Numerik
        kolom_numerik = ['Umur','NilaiBelanjaSetahun']  
        
        # Statistik sebelum Standardisasi  
        print('Statistik Sebelum Standardisasi\n')  
        print(df[kolom_numerik].describe().round(1))

        # Standardisasi  
        df_std = StandardScaler().fit_transform(df[kolom_numerik])

        # Membuat DataFrame  
        df_std = pd.DataFrame(data=df_std, index=df.index, columns=df[kolom_numerik].columns)

        # Menampilkan contoh isi data dan summary statistic
        print('Contoh hasil standardisasi\n') 
        print(df_std.head())
        
        print('Statistik hasil standardisasi\n')
        print(df_std.describe().round(0))

        # Konversi Kategorikal Data                
        # Inisiasi nama kolom kategorikal  
        kolom_kategorikal = ['Jenis Kelamin','Profesi','Tipe Residen']  
        
        # Membuat salinan data frame  
        df_encode = df[kolom_kategorikal].copy()  
        
        # Melakukan labelEncoder untuk semua kolom kategorikal  
        for col in kolom_kategorikal:  
            df_encode[col]= LabelEncoder().fit_transform(df_encode[col])
            
        # Menampilkan data  
        print(df_encode.head())
        
        # Menggabungkan data frame
        df_model = df_encode.merge(df_std, left_index = True, right_index=True, how= 'left')  
        print (df_model.head())


data = "https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/customer_segments.txt"

app = Cust_segment()
data_raw = app.read_data()
app.eksplorasi_dataNumerik(data_raw)
app.eksplorasi_dataKategorikal(data_raw)
final_data = app.preparation_data(data_raw)