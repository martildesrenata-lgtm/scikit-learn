import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Criando uma base de dados fictícia de uma empresa
data = {
    'meses_contrato': [1, 12, 24, 2, 36, 48, 3, 10, 5, 20],
    'valor_mensal': [100, 80, 70, 110, 60, 55, 95, 85, 105, 75],
    'suporte_acionado': [5, 1, 0, 4, 0, 0, 6, 2, 4, 1],
    'cancelou': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0] # 1 = Sim, 0 = Não
}

df = pd.DataFrame(data)

# 2. Separando o que queremos prever (cancelou) das características (features)
X = df.drop('cancelou', axis=1)
y = df['cancelou']

# 3. Treinando o modelo de IA (Random Forest)
modelo = RandomForestClassifier()
modelo.fit(X, y)

# 4. Simulando um novo cliente para a IA decidir
# Cliente: 2 meses de contrato, paga 120, acionou suporte 7 vezes
novo_cliente = [[2, 120, 7]]
previsao = modelo.predict(novo_cliente)

print(f"Previsão da IA para o novo cliente: {'Vai Cancelar' if previsao[0] == 1 else 'Vai Ficar'}")
