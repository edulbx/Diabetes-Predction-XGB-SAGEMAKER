## Model XGBOOST and Endpoint for Deploy

## Imports


```python
# Imports
import os
import json
import sagemaker
import boto3
import numpy as np
import pandas as pd
from sagemaker.serializers import CSVSerializer
from sagemaker.inputs import TrainingInput
from sagemaker.predictor import Predictor
from sagemaker import get_execution_role
```

## Carrega os Dados


```python
# Obtém a sessão do SageMaker
session = boto3.Session()
```


```python
s3 = session.resource('s3')
```


```python
s3
```


```python
from sagemaker import get_execution_role
role = get_execution_role()
print(role)
```


```python
# Altere para o nome do seu bucket
s3_bucket = 'eduardo-project-medical-data'
prefix = 'dados'
```


```python
raiz = 's3://{}/{}/'.format(s3_bucket, prefix)
print(raiz)
```


```python
dados_treino = TrainingInput(s3_data = raiz + 'treino.csv', content_type = 'csv')
dados_teste = TrainingInput(s3_data = raiz + 'teste.csv', content_type = 'csv')
```


```python
print(json.dumps(dados_treino.__dict__, indent = 2))
```


```python
print(json.dumps(dados_teste.__dict__, indent = 2))
```

## Construção e Treinamento do Modelo XGB


```python

container_uri = sagemaker.image_uris.retrieve(region = session.region_name,
                                              framework = 'xgboost',
                                              version = '1.0-1',
                                              image_scope = 'training')
```

Criação do Container - seguindo a documentação da AWS estou criando um container para podermos usar apenas para treinamento uma máquina mais
potente disponível na versão gratuita do SageMaker. Para os parametros da função abaixo estou recuperando os dados da região utilizando o BOTO, os demais
parametros são ajustaveis, mas é necessário conferir os custos.


```python
# Argumentos do estimador para serem usados na função de criação
sagemaker_execution_role = role
sagemaker_session = sagemaker.Session()
```


```python
# Criação do Estimador - estou seguindo a documentação para a criação.
xgb = sagemaker.estimator.Estimator(image_uri = container_uri,
                                    role = sagemaker_execution_role,
                                    instance_count = 2,
                                    instance_type = 'ml.m5.large', #note que essa máquina/instancia tem apenas 50 hrs no nível gratuito.
                                    output_path = 's3://{}/artefatos'.format(s3_bucket),
                                    sagemaker_session = sagemaker_session,
                                    base_job_name = 'classifier')
```


```python
# Definição dos Hiperparâmetros - consultar a documenação caso queira mudar
xgb.set_hyperparameters(objective = 'binary:logistic', num_round = 100)
```


```python
# Treinamento do modelo
xgb.fit({'train': dados_treino, 'validation': dados_teste})
```

## Gerando o Endpoint a Partir do Modelo


```python

xgb_predictor = xgb.deploy(initial_instance_count = 2, instance_type = 'ml.m5.large')
```

Deploy do modelo treinado criando o endpoint o .deploy ajusta o resultado para que possa ser usado os resultados para outros propositos, inclusive o deploy
e dessa forma salvamos o modelo para ser usado por aplicações.

## Previsões a Partir do Endpoint


```python
csv_serializer = CSVSerializer()
```


```python
predictor = Predictor(endpoint_name = xgb_predictor.endpoint_name, serializer = csv_serializer)
```


```python
df_teste = pd.read_csv(raiz + 'teste.csv', names = ['class', 'bmi', 'diastolic_bp_change', 'systolic_bp_change', 'respiratory_rate'])
```


```python
df_teste.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>bmi</th>
      <th>diastolic_bp_change</th>
      <th>systolic_bp_change</th>
      <th>respiratory_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-0.940089</td>
      <td>-0.403964</td>
      <td>-0.279542</td>
      <td>-0.817379</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>-0.502614</td>
      <td>-0.665582</td>
      <td>0.131742</td>
      <td>-0.362450</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1.078473</td>
      <td>0.347981</td>
      <td>0.228029</td>
      <td>-0.817379</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>-0.636164</td>
      <td>-0.251491</td>
      <td>0.587034</td>
      <td>-0.817379</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>-0.528479</td>
      <td>2.037253</td>
      <td>1.383463</td>
      <td>0.185934</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = df_teste.sample(1)
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>bmi</th>
      <th>diastolic_bp_change</th>
      <th>systolic_bp_change</th>
      <th>respiratory_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1017</th>
      <td>0</td>
      <td>-0.022864</td>
      <td>-0.496655</td>
      <td>2.153753</td>
      <td>-0.067314</td>
    </tr>
  </tbody>
</table>
</div>




```python
X = X.values[0]
X[1:]
```




    array([-0.02286428, -0.49665455,  2.15375335, -0.06731361])




```python
paciente = X[1:]
paciente
```




    array([-0.02286428, -0.49665455,  2.15375335, -0.06731361])




```python
# Faz a previsão de um paciente
predicted_class_prob = predictor.predict(paciente).decode('utf-8')
if float(predicted_class_prob) < 0.5:
    print('Previsão = Não Diabético')
else:
    print('Previsão = Diabético')
print()
```

    Previsão = Não Diabético
    
    

## Avaliando o Modelo


```python
# Previsão de todos os pacientes no dataset de teste
predictions = []
expected = []
correct = 0
for row in df_teste.values:
    expected_class = row[0]
    payload = row[1:]
    predicted_class_prob = predictor.predict(payload).decode('utf-8')
    predicted_class = 1
    if float(predicted_class_prob) < 0.5:
        predicted_class = 0
    if predicted_class == expected_class:
        correct += 1
    predictions.append(predicted_class)
    expected.append(expected_class)
```


```python
print('Acurácia = {:.2f}%'.format(correct/len(predictions) * 100))
```

    Acurácia = 77.72%
    

#### Confusion Matrix


```python
expected = pd.Series(np.array(expected))
predictions = pd.Series(np.array(predictions))
pd.crosstab(expected, predictions, rownames = ['Actual'], colnames = ['Predicted'], margins = True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Predicted</th>
      <th>0</th>
      <th>1</th>
      <th>All</th>
    </tr>
    <tr>
      <th>Actual</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>1909</td>
      <td>71</td>
      <td>1980</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>483</td>
      <td>24</td>
      <td>507</td>
    </tr>
    <tr>
      <th>All</th>
      <td>2392</td>
      <td>95</td>
      <td>2487</td>
    </tr>
  </tbody>
</table>
</div>



**That's all folks**


```python
!jupyter nbconvert --to markdown Model_XGBoost_Training_and_Endpoint.ipynb
```


```python
!pip install nbconvert
```

    Defaulting to user installation because normal site-packages is not writeable
    Collecting nbconvert
      Downloading nbconvert-7.16.4-py3-none-any.whl.metadata (8.5 kB)
    Requirement already satisfied: beautifulsoup4 in c:\users\edu\appdata\roaming\python\python312\site-packages (from nbconvert) (4.12.3)
    Collecting bleach!=5.0.0 (from nbconvert)
      Downloading bleach-6.1.0-py3-none-any.whl.metadata (30 kB)
    Collecting defusedxml (from nbconvert)
      Downloading defusedxml-0.7.1-py2.py3-none-any.whl.metadata (32 kB)
    Requirement already satisfied: jinja2>=3.0 in c:\users\edu\appdata\roaming\python\python312\site-packages (from nbconvert) (3.1.4)
    Requirement already satisfied: jupyter-core>=4.7 in c:\users\edu\appdata\roaming\python\python312\site-packages (from nbconvert) (5.7.2)
    Collecting jupyterlab-pygments (from nbconvert)
      Downloading jupyterlab_pygments-0.3.0-py3-none-any.whl.metadata (4.4 kB)
    Requirement already satisfied: markupsafe>=2.0 in c:\users\edu\appdata\roaming\python\python312\site-packages (from nbconvert) (2.1.5)
    Collecting mistune<4,>=2.0.3 (from nbconvert)
      Downloading mistune-3.0.2-py3-none-any.whl.metadata (1.7 kB)
    Collecting nbclient>=0.5.0 (from nbconvert)
      Downloading nbclient-0.10.0-py3-none-any.whl.metadata (7.8 kB)
    Collecting nbformat>=5.7 (from nbconvert)
      Downloading nbformat-5.10.4-py3-none-any.whl.metadata (3.6 kB)
    Requirement already satisfied: packaging in c:\users\edu\appdata\roaming\python\python312\site-packages (from nbconvert) (24.1)
    Collecting pandocfilters>=1.4.1 (from nbconvert)
      Downloading pandocfilters-1.5.1-py2.py3-none-any.whl.metadata (9.0 kB)
    Requirement already satisfied: pygments>=2.4.1 in c:\users\edu\appdata\roaming\python\python312\site-packages (from nbconvert) (2.18.0)
    Collecting tinycss2 (from nbconvert)
      Downloading tinycss2-1.3.0-py3-none-any.whl.metadata (3.0 kB)
    Requirement already satisfied: traitlets>=5.1 in c:\users\edu\appdata\roaming\python\python312\site-packages (from nbconvert) (5.14.3)
    Requirement already satisfied: six>=1.9.0 in c:\users\edu\appdata\roaming\python\python312\site-packages (from bleach!=5.0.0->nbconvert) (1.16.0)
    Collecting webencodings (from bleach!=5.0.0->nbconvert)
      Downloading webencodings-0.5.1-py2.py3-none-any.whl.metadata (2.1 kB)
    Requirement already satisfied: platformdirs>=2.5 in c:\users\edu\appdata\roaming\python\python312\site-packages (from jupyter-core>=4.7->nbconvert) (4.3.2)
    Requirement already satisfied: pywin32>=300 in c:\users\edu\appdata\roaming\python\python312\site-packages (from jupyter-core>=4.7->nbconvert) (306)
    Requirement already satisfied: jupyter-client>=6.1.12 in c:\users\edu\appdata\roaming\python\python312\site-packages (from nbclient>=0.5.0->nbconvert) (8.6.2)
    Collecting fastjsonschema>=2.15 (from nbformat>=5.7->nbconvert)
      Downloading fastjsonschema-2.20.0-py3-none-any.whl.metadata (2.1 kB)
    Requirement already satisfied: jsonschema>=2.6 in c:\users\edu\appdata\roaming\python\python312\site-packages (from nbformat>=5.7->nbconvert) (4.23.0)
    Requirement already satisfied: soupsieve>1.2 in c:\users\edu\appdata\roaming\python\python312\site-packages (from beautifulsoup4->nbconvert) (2.6)
    Requirement already satisfied: attrs>=22.2.0 in c:\users\edu\appdata\roaming\python\python312\site-packages (from jsonschema>=2.6->nbformat>=5.7->nbconvert) (24.2.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\users\edu\appdata\roaming\python\python312\site-packages (from jsonschema>=2.6->nbformat>=5.7->nbconvert) (2023.12.1)
    Requirement already satisfied: referencing>=0.28.4 in c:\users\edu\appdata\roaming\python\python312\site-packages (from jsonschema>=2.6->nbformat>=5.7->nbconvert) (0.35.1)
    Requirement already satisfied: rpds-py>=0.7.1 in c:\users\edu\appdata\roaming\python\python312\site-packages (from jsonschema>=2.6->nbformat>=5.7->nbconvert) (0.20.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\edu\appdata\roaming\python\python312\site-packages (from jupyter-client>=6.1.12->nbclient>=0.5.0->nbconvert) (2.9.0.post0)
    Requirement already satisfied: pyzmq>=23.0 in c:\users\edu\appdata\roaming\python\python312\site-packages (from jupyter-client>=6.1.12->nbclient>=0.5.0->nbconvert) (26.2.0)
    Requirement already satisfied: tornado>=6.2 in c:\users\edu\appdata\roaming\python\python312\site-packages (from jupyter-client>=6.1.12->nbclient>=0.5.0->nbconvert) (6.4.1)
    Downloading nbconvert-7.16.4-py3-none-any.whl (257 kB)
    Downloading bleach-6.1.0-py3-none-any.whl (162 kB)
    Downloading mistune-3.0.2-py3-none-any.whl (47 kB)
    Downloading nbclient-0.10.0-py3-none-any.whl (25 kB)
    Downloading nbformat-5.10.4-py3-none-any.whl (78 kB)
    Downloading pandocfilters-1.5.1-py2.py3-none-any.whl (8.7 kB)
    Downloading defusedxml-0.7.1-py2.py3-none-any.whl (25 kB)
    Downloading jupyterlab_pygments-0.3.0-py3-none-any.whl (15 kB)
    Downloading tinycss2-1.3.0-py3-none-any.whl (22 kB)
    Downloading fastjsonschema-2.20.0-py3-none-any.whl (23 kB)
    Downloading webencodings-0.5.1-py2.py3-none-any.whl (11 kB)
    Installing collected packages: webencodings, fastjsonschema, tinycss2, pandocfilters, mistune, jupyterlab-pygments, defusedxml, bleach, nbformat, nbclient, nbconvert
    Successfully installed bleach-6.1.0 defusedxml-0.7.1 fastjsonschema-2.20.0 jupyterlab-pygments-0.3.0 mistune-3.0.2 nbclient-0.10.0 nbconvert-7.16.4 nbformat-5.10.4 pandocfilters-1.5.1 tinycss2-1.3.0 webencodings-0.5.1
    
