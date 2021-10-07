import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from math import cos, pi


def himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


class Functions_utils:

    def __init__(self, train_row, teste_row, columns) -> None:
        self.dataset = pd.read_csv(r'ransomware.csv')

        self.tipo = self.dataset.iloc[:train_row, 1]
        self.dados = self.dataset.iloc[:train_row, 3:columns+3]

        self.tipo_teste = self.dataset.iloc[teste_row:, 1].reset_index(drop=True)
        self.dados_teste = self.dataset.iloc[teste_row:, 3:columns+3].reset_index(drop=True)

        self.range = 0.5
    
    def pca(self, dim):
        pca = PCA(n_components=dim)
        self.dados = pd.DataFrame(pca.fit_transform(self.dados.values))
        self.dados_teste = pd.DataFrame(pca.transform(self.dados_teste.values))
        self.range = 0

    def sigmoid(self, x):
        return 1 if x>self.range else 0

    def compare(self, regras_particula, row_dataset):
        count = 0
        for i in range(len(regras_particula)):
            if row_dataset[i] == self.sigmoid(regras_particula[i]):
                count+=1
            else:
                count-=1
        return 1 if count>=0 else 0


    def sensibilidade_especifidade(self, particula):
        tp,fp,fn,tn = 0, 0, 0, 0
        for c in range(len(self.dados)):
            predicao = self.compare(particula, self.dados.iloc[c])

            if predicao==1:
                if self.tipo[c]==1:
                    tp+=1
                elif self.tipo[c]==0:
                    fp+=1
            elif predicao==0:
                if self.tipo[c]==1:
                    fn+=1
                elif self.tipo[c]==0:
                    tn+=1

        sensibilidade = round(tp/(tp+fn),3)
        especifidade = round(tn/(tn+fp),3)

        #return 1 - ((tp+tn)/(tp+fn+tn+fp))
        return 1-(sensibilidade*especifidade)  

    def sensibilidade_especifidade2(self, particula):
        tp,fp,fn,tn = 0, 0, 0, 0
        for c in range(len(self.dados)):
            predicao = self.compare(particula, self.dados.iloc[c])

            if predicao==1:
                if self.tipo[c]==1:
                    tp+=1
                elif self.tipo[c]==0:
                    fp+=1
            elif predicao==0:
                if self.tipo[c]==1:
                    fn+=1
                elif self.tipo[c]==0:
                    tn+=1

        sensibilidade = round(tp/(tp+fn),3)
        especifidade = round(tn/(tn+fp),3)

        return tp,fp,fn,tn,sensibilidade,especifidade

    def sensibilidade_especifidade3(self, particula):
        tp,fp,fn,tn = 0, 0, 0, 0
        for c in range(len(self.dados_teste)):
            predicao = self.compare(particula, self.dados_teste.iloc[c])

            if predicao==1:
                if self.tipo_teste[c]==1:
                    tp+=1
                elif self.tipo_teste[c]==0:
                    fp+=1
            elif predicao==0:
                if self.tipo_teste[c]==1:
                    fn+=1
                elif self.tipo_teste[c]==0:
                    tn+=1

        sensibilidade = round(tp/(tp+fn),3)
        especifidade = round(tn/(tn+fp),3)

        return tp,fp,fn,tn,sensibilidade,especifidade 

    def plot_confusion_matrix(self, tp,fp,fn,tn):
        ax = plt.subplot()
        sns.heatmap([[tp, fn],[fp, tn]], annot=True, ax = ax)

        # labels, title and ticks
        ax.set_xlabel('Valor Predito')
        ax.set_ylabel('Real')
        ax.set_title('Matriz de confusão')
        ax.xaxis.set_ticklabels(['Ransomware', 'Goodware']); ax.yaxis.set_ticklabels(['Ransomware', 'Goodware'])
        
  
