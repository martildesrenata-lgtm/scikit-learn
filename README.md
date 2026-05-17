import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# 1. Simulando dados: O que afeta a retenção de clientes?
data = {
    'Investimento_Treinamento': [10, 20, 5, 25, 30, 12, 18, 5],
    'Uso_de_Ferramentas_IA': [80, 90, 20, 95, 100, 40, 70, 10],
    'Velocidade_Atendimento': [95, 98, 30, 99, 100, 50, 85, 20],
    'Lucratividade': [1, 1, 0, 1, 1, 0, 1, 0] # 1 = Alta, 0 = Baixa
}

df = pd.DataFrame(data)
X = df.drop('Lucratividade', axis=1)
y = df['Lucratividade']

# 2. Treinando a IA para entender a importância de cada fator
model = RandomForestClassifier()
model.fit(X, y)

# 3. Criando o gráfico de importância para a empresa
importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(10, 5))
sns.barplot(x=importances.values, y=importances.index, palette='magma')
plt.title('Impacto da IA e Treinamento na Lucratividade da Empresa')
plt.xlabel('Grau de Importância (0 a 1)')
plt.show()
