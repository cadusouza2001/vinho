# Predição da Qualidade de Vinhos com Redes Neurais

## Integrantes do Grupo
Carlos Souza

Gustavo Parcianello Cardona

Murilo Schuck


## 1. Descrição do Problema
Este projeto visa desenvolver um modelo de rede neural capaz de prever a qualidade de vinhos (tintos e brancos) com base em suas propriedades físico-químicas. A qualidade do vinho é uma avaliação subjetiva, mas o dataset utilizado a representa com uma nota de 3 a 9, atribuída por especialistas.

A relevância desta aplicação está no potencial de auxiliar produtores de vinho no controle de qualidade, na otimização de processos de produção e na segmentação de mercado, permitindo uma precificação mais assertiva baseada em dados objetivos.


## 2. Dataset Utilizado
O conjunto de dados combina amostras de vinhos tintos e brancos, contendo 12 atributos de entrada (features) e 1 variável de saída (a nota de qualidade).

Features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol, type.
Alvo (Target): quality (nota de 3 a 9).


## 3. Abordagens e Arquiteturas dos Modelos
Neste projeto, exploramos duas abordagens distintas para resolver o problema, cada uma com um modelo de rede neural específico.

### Abordagem 1: Modelo de Regressão (wine_quality_reg.ipynb)
A primeira tentativa foi tratar o problema como uma regressão, onde o objetivo era prever a nota exata da qualidade do vinho (um valor contínuo de 3 a 9).

Resultado e Análise Crítica:
O modelo treinou, mas o desempenho foi insatisfatório. A métrica R² (Coeficiente de Determinação) foi de apenas 0.338, o que significa que o modelo só conseguia explicar cerca de 33,8% da variabilidade na qualidade do vinho. O Erro Absoluto Médio (MAE) de 0.547 indicou que, em média, as previsões erravam a nota em mais de meio ponto. Concluímos que prever a nota exata era uma tarefa muito difícil com os dados disponíveis.

### Abordagem 2: Modelo de Classificação (wine_classification_model.ipynb)
Diante do baixo desempenho da regressão, decidimos fazer um modelo de classificação também, com o objetivo de lassificar os vinhos em três categorias: Ruim (nota ≤ 4), Médio (nota 5-7) e Bom (nota ≥ 8).

Resultado e Análise Crítica:
À primeira vista, o desempenho do modelo de classificação pareceu um grande avanço, com uma acurácia geral de 82%. No entanto, uma análise mais detalhada das métricas por classe revelou um desempenho bastante desbalanceado. O F1-score para a classe "Média" foi excelente, atingindo 0.90, o que mostra que o modelo é muito eficaz em identificar a classe majoritária. Em contrapartida, o desempenho nas classes minoritárias foi muito baixo, com F1-scores de apenas 0.26 para vinhos "Ruins" e 0.32 para "Bons".

Isso nos levou à conclusão de que a alta acurácia é uma métrica enganosa neste caso. O modelo, na prática, aprendeu a "apostar" na classe mais frequente, mas não conseguiu capturar os padrões complexos que definem vinhos de qualidade ruim ou excelente, tornando-o pouco útil para uma análise de qualidade criteriosa.

## 4. Conclusão 
A alta acurácia geral é enganosa. Ela é impulsionada pelo fato de o dataset ser extremamente desbalanceado, com a grande maioria das amostras estar localizada na região média.
