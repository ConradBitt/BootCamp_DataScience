# BootCamp_DataScience
Repositório para armazenar as anotações em notebooks python e os relatórios sobre o BootCamp DataScience 2020.

# Resumo do trabalho final

Neste trabalho foram utilizados quatro estimadosres, **Logistic Regression, Decision Tree Classifier, SGDClassifier** e **SVC**, além disso um base line (DummyClassifier), cujo objetivo principal é classificar se pacientes precisam ou não de um serviço de atendimento em um dos leitos da unidade de terapia intensiva. Dentre as mais de duzentas variáveis disponíveis para utilizar neste ofício, foi feita uma seleção através de testes não paramétricos para garantir que as variáveis não provenham de uma mesma distribuição, em seguida fizemos uma busca aleatória por hiperparâmetros dos modelos com a técnica de validação cruzada. Por fim, os objetivos foram atingidos com a ressalva de que apenas um dos modelos, **o Logistic Regression, mostrou com a vantagem de errar menos do que os outros, apresentando uma sensibilidade de 70% e 73% entre os pacientes que precisam e não precisam de UTI, respectivamente, e um valor preditivo positivo de 80% para os pacientes que não precisam de UTI e 26% de falsos negativos para a mesma situação**.

# Descobertas
## Sobre análise das variáveis disponíveis

Dentre as variáveis, 255 são do tipo float64, 4 variáveis do tipo int64 e 2 do tipo object que são do tipo string. **Outra informação que é relevante é a quantidade de variáveis do tipo média, mediana, máximo, mínimo, diff (Máximo - Mínimo) e diff relativa (diff/mediana)**, fato dessas variáveis terem dependência pode gerar uma correlação e isso motivou a construção de uma mapa de correlação entre cada uma das variáveis.

<img src='https://raw.githubusercontent.com/ConradBitt/BootCamp_DataScience/master/projeto_final/imagens_para_resumo/Correla%C3%A7%C3%A3o.png'>

## A respeito da normalidade das variaveis e a decisão de aplicar testes não paramétricos 

Eu pensei em aplicar técnicas de análise de variância, entretanto muitas das variáveis foram construidas uma dependendo da outra. As exigências da ANOVA é que as amostras sejam independentes, que suas populações sejam distribuídas normalmente e que as variâncias populacionais sejam iguais. **O que não se verifica por um teste de normalidade, por isso decidi aplicar um teste não paramétrico de Wilcoxon ao conjunto de variáveis e verificar quais provém de uma mesma distribuição ou não**.

<img src='https://raw.githubusercontent.com/ConradBitt/BootCamp_DataScience/master/projeto_final/imagens_para_resumo/wilcoxon_test.png'>

No mapa de Wilcoxon podemos ver quais variáveis não tem a mesma distribuição, são todas as partes em coloração azul escuro, isto porque são todas as partes em que o p-valor ≤ 0.05, acima disso não se tem evidência estatística para rejeitar a hipótese de que as variáveis provém de uma mesma distribuição.
Note que uma das condições é que as variáveis sejam aleatórias independentes, portanto algumas variáveis certamente não se tem evidência para rejeitar a hipótese nula, pois foram construidas em função de outras, existe uma dependência linear entre as variáveis, isso é possível notar na diagonal cujo o p-valor é aproximadamente 1. Desta forma, dentre as 230 variáveis reduzimos o problema à 52. 

<img src='https://raw.githubusercontent.com/ConradBitt/BootCamp_DataScience/master/projeto_final/imagens_para_resumo/variaveis.png'>

Com certeza, uma das variáveis que chamam bastante a atenção são `AGE_ABOVE65` que indica se o paciente tem mais de 65 anos, o que é um fator bem importante, pois até a coleta desses dados a variante que estava em circulação no Brasil afetava significativamente pessoas mais velhas. Além disso outras variáveis que chamam a atenção são a creatina média, linfócitos, neutrófilos e ureia. Existem alguns artigos publicados que chegaram à resultados parecidos, onde estas variáveis citadas se mostram importantes para a classificação de um paciente em um estado clinico de covid-19.

> [Machine Learning Approach to Predicting COVID-19 Disease Severity Based on Clinical Blood Test Data: Statistical Analysis and Model Development](https://doi.org/10.2196/25884)

> [Crucial laboratory parameters in COVID-19 diagnosis and prognosis: An updated meta-analysis](http://dx.doi.org/10.1016/j.medcli.2020.05.017)

> [El test rápido de inmunoglobulinas confirma un caso sospechoso de COVID-19](http://dx.doi.org/10.1016/j.medcli.2020.04.008)

## Definindo a separação das variáveis e validação cruzada

Vou utilizar a **técnica de cross-validation, ela consiste etapas consecutivas de treino e teste em conjuntos selecionados aleatoriamente dentre os dados possíveis**. Na primeira etapa o algorítmo vai separar a base de dados em conjuntos de treino e teste, então treina e testa. Depois disso ele passa para a próxima iteração onde separará outro conjunto de treino e teste até que passe por todo o conjunto de dados.

Entretanto **pode acontecer um problema ao utilizar a validação cruzada: o overfitting**. Ele ocorre quando o modelo fica extremamente eficiente aos dados de treino e teste utilizado na validação cruzada, mas não consegue generalizar as informações fora desse conjunto. A ideia então é separar um outro conjunto de dados que o modelo não vai utilizar na validação cruzada, esse conjunto é chamado de dados de validação, como mostra a imagem de todos os dados. Como ilustra a imagem abaixo:

<img src='https://raw.githubusercontent.com/ConradBitt/BootCamp_DataScience/master/projeto_final/imagens_para_resumo/all_data_split_validation_and_cross_validation.png'>

É comum encontrar uma proporção de 25% de dados de treino e 75% em dados de teste. Vou separar 20% dos dados totais para validação dos modelos e dos 80% restante os dados de treino e teste. 80% dos dados reservados para treino e teste, 80% serão dados de treino e 20% dados de teste. Portanto no geral 20% serão dados de validação, 64%% dados de treino e 16% de teste.

Apesar dos grupos serem sempre estratificados de acordo com a proporção de amostras que foram para a UTI, a discussão a respeito dessas proporções de treino, teste e validação é completamente pertinente e o fato de ter poucos dados de treino e teste motiva o uso da validação cruzada.

## Sobre as pontuações dos estimadores

Temos os resultados de cada curva ROC, **que oferece uma relação entre falsos positivos e falsos negativos**, dos estimadores que estamos testando para tentar resolver o problema da classificação de pacientes. Todos os estimadores tem um desempenho acima do base line estipulado pelo dummy classifier indicado pela linha vermelha tracejada. Ou seja, pela análise gráfica temos um resultado, no mínimo melhor que Dummy Classifier.

<img src='https://raw.githubusercontent.com/ConradBitt/BootCamp_DataScience/master/projeto_final/imagens_para_resumo/curva_ROC.png'>

É de se esperar que a curva ROC dos dados de teste (em azul) tenham um desempenho maior do que nos dados de validação (em laranja), isso porque na validação cruzada inevitavelmente o modelo irá entrar em contato com alguns elementos repetidos. Entretanto os modelos conseguiram generalizar ao ponto de ter um desempenho acima dos 0.7 mesmo em dados que nunca tinham visto antes que foram separados anteriormente. **Em alguns casos, por exemplo o do estimador Logistic Regression a curva de validação chega bem próxima da dos dados de teste, isto é em dados que o modelo nunca tinha visto antes**.

## Análise da Matriz de confusão dos estimadores

Na matriz de confusão fica mais fácil de avaliar a qualidade dos modelos. Os dados estão normalizados pelas linhas, isto é, pelos valores reais, não pelos preditos. Também deixarei disponível uma tabela com os valores do preditivo positivo e da sensibilidade de cada modelo.

<img src='https://raw.githubusercontent.com/ConradBitt/BootCamp_DataScience/master/projeto_final/imagens_para_resumo/matriz%2Bde%2Bconfus%C3%A3o.png'>

No gráfico do modelo Logistic Regression, podemos concluir que de todos os pacientes que não precisam da UTI o modelo acertou corretamente 73% e 27% são falsos negativos, pessoas que não precisavam do serviço da UTI mas o modelo considerou que precisavam. Além disso, para os pacientes que realmente precisvam do serviço de UTI o modelo classificou corretamente 70% deles, entretanto 30% foram falsos positivos, ou seja, pessoas que precisavam do serviço mas não foram encaminhadas.

O modelo de árvore de decisão (Decision Tree Classifier) classificou, dentre os pacientes que não precisavam de UTI, 93% corretamente e a quantidade de falsos negativos foi de 7%. Por outro lado a quantidade de falsos positivos deste modelo foi de relativamente alta, dentre as pessoas que precisavam de UTI ele classificou que 73% não precisavam e classificou corretamente apenas 27%.

A classificação dos dois modelos, stochastic gradient descent e suport vector classification, foram péssimas em relação aos demais. Para resolver o problema estes modelos simplesmente classificaram que todos os pacientes não precisam de UTI. Para concensar todoas essas informações temos as métricas de avaliação: 

<img src='https://raw.githubusercontent.com/ConradBitt/BootCamp_DataScience/master/projeto_final/imagens_para_resumo/classification_report.png'>

Tendo em vista que nosso objetivo é determinar quais pacientes precisam ir ou não para o serviço da unidade de terapia intensiva todo os modelos não tiveram um bom desempenho, pois se tratando de serviços de saúde os modelos SGDClassifier, SVC e DecisionTreeClassifier, com os parâmetros que eu utilizei não devem ser postos em produção em nenhuma hipótese, tendo em vista o custo de um falso positivo ser muito alto. Apesar do modelo Logistic Regression ter a menor proporção de falsos positivos dentre os modelos testados, ainda sim é um valor alto. O que pode ser feito é tentar aprimorar esse modelo.


## Dados brutos


