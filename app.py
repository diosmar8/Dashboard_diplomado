from ast import If
import streamlit as st 
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import  GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import (f1_score, roc_auc_score, precision_recall_curve, 
                            roc_curve, confusion_matrix, classification_report, 
                            accuracy_score)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.neighbors import KNeighborsClassifier


# Titulos del aplicativo 
st.title("HOMICIDIOS EN COLOMBIA") 
st.sidebar.write("HOMICIDIOS OCURRIDOS EN COLOMBIA DESDE EL AÑO 2010 HASTA EL 2021")
st.sidebar.write("------")

# Leemos los datos procesados 
datos=pd.read_csv('datosLimpio.csv')
#creamos una columna para el numero del mes 
datos['mes_num'] =[1 if (f=='Enero') else 2 if (f=='Febrero') else 3 if (f=='Marzo') else 4 if f=='Abril'
 else 5 if f=='Mayo' else 6 if f=='Junio' else 7 if f=='Julio' else 8 if f=='Agosto' 
 else 9 if f=='Septiembre' else 10 if f=='Octubre' else 11 if f=='Noviembre' else 12
 for f in datos['mes']]

# declaramos las variables 
departa=list(datos['departamento'].unique())
departamento= st.sidebar.selectbox(label='SELECCIONE UN DEPARTAMENTO', options=departa)
# filtrar por departamento Obtenemos una lista del departamento dado 

datosfiltroDepar=datos[datos['departamento']==departamento]
st.write("LISTA DE HOMICIDIOS DEL DEPARTAMENTO  DE "+departamento)
st.write(datosfiltroDepar.drop(columns='codigo'))


st.sidebar.write("------")
# filtrar por el municipios seleccionado 
municipio=list(datosfiltroDepar['municipio'].unique())
selecMunicipio= st.sidebar.selectbox(label='SELECCIONE UN MUNICIPIO', options=municipio)
datosFiltroMunicipio=datosfiltroDepar[datosfiltroDepar['municipio']==selecMunicipio]
st.write("------")
st.write("LISTA DE HOMICIDIOS DEL MUNICIPIO "+selecMunicipio+" UBICADO EN EL DEPARTAMENTO "+departamento)
st.write(datosFiltroMunicipio.drop(columns='codigo'))

# graficas para ver la cantdad de homicidios por municipios  de un departamento dado    
fig_d = px.bar(datosfiltroDepar, x='municipio', y='cantidad',
                labels={'Intervalos': 'Mes'},height=600,width=900,
                color_discrete_sequence=px.colors.sequential.Aggrnyl,
                 title = "GRAFICA REPRESENTATIVA DE LOS MUNICIPIOS DEL DEPARTAMENTO "+departamento)
st.plotly_chart(fig_d)

## GRAFICA PARA VER LOS HOMICIDIOS POR AÑOS PARA EL MUNICIPIO SELECCIONADO  
st.write("------")
fig = px.bar(datosfiltroDepar, x='año', y='cantidad',
                labels={'Intervalos': 'Mes'},height=600,width=900,
                color_discrete_sequence=px.colors.sequential.Aggrnyl,
                 title = "GRAFICA REPRESENTATIVA DE LOS MUNICIPIOS DEL DEPARTAMENTO "+departamento+" POR CADA AÑO")
st.plotly_chart(fig)



# calculamos un rango de años 
st.sidebar.write("------")
selecAnos= st.sidebar.selectbox(label='SELECCIONE UN AÑO', options=['todos',2010,2011,2012,2013,2014,2015,2016
  ,2017,2018,2019,2020,2021])

if(selecAnos=='todos'):
    rango_anos=datosFiltroMunicipio
    textoano='PARA TODO LOS AÑOS'
else:
   rango_anos=datosFiltroMunicipio[datosFiltroMunicipio['año']==selecAnos]
   textoano='PARA EL AÑO SELECCIONADO'

st.sidebar.write("------")
selecmes= st.sidebar.selectbox(label='SELECCIONE UN MES', options=['todos','Enero','Febrero','Marzo',
'Abrir','Mayo','Junio','Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre'])

if(selecmes=='todos'):
    rango_mes=rango_anos
else:
   rango_mes=rango_anos[rango_anos['mes']==selecmes]


st.sidebar.write("------")
# Elejimos la columna para el grafico torta 
seleccolumnas= st.sidebar.selectbox(label='SELECCIONE POR GENERO O POR EDAD DE GRUPO ', options=['genero','edad_grupo'])
fig_B = px.pie(rango_mes.reset_index(),
 values='cantidad', 
 names=seleccolumnas,
  title='DISTRIBUCION POR  '+ seleccolumnas+' DEL MUNICIPIO '+selecMunicipio+" "+textoano,
  color_discrete_sequence=px.colors.sequential.Aggrnyl)
st.plotly_chart(fig_B)



# Grafica para saber que tipo de arma se usa mas 
fig_arma = px.pie(rango_mes.reset_index(),
 values='cantidad',
  names='arma',
  color_discrete_sequence=px.colors.sequential.Aggrnyl,
               title='DISTRIBUCION POR ARMAS PARA EL MUNICIPIO '+selecMunicipio)
st.plotly_chart(fig_arma)

# Grafico para ver el mes com mas homicidios 


fig_Mes = px.bar(rango_mes, x='mes', y='cantidad',
                labels={'Intervalos': 'Mes'},
                height=600,
                width=900,
                color_discrete_sequence=px.colors.sequential.Aggrnyl,
                 title = "REPRESENTACION DE LOS HOMICIDIOS"+
                 " PARA EL MUNICIPIO "+selecMunicipio+" POR MESES")
st.plotly_chart(fig_Mes)






## CODIGO PARA EL MIDELO 

data = rango_mes[['departamento','municipio','año','mes','dia_mes','arma','genero','edad_grupo']].copy()

data_to_model=pd.get_dummies(data, drop_first=True)
data.groupby(['departamento','municipio','año','mes','dia_mes','arma','genero','edad_grupo']).sum().reset_index()

def get_X_y(df, y_name):

  y=[y_name]
  X=[col for col in df.columns if col not in y]
  y=df[y].copy().values.flatten()
  X=pd.get_dummies(df[X].copy())
  return X, y


def data_preprocessing_up_or_down_sample(X, y, sample="up", test_size=0.2):
 
  a,b=0,0
  if sample=="up": 
    a,b=1,0
  if sample=="down":
    a,b=0,1 

  
  # Apply the normal train_test_split to the data
  X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size)

  if a+b>=1:
    X_train_temp, y_train_temp = resample(X_train[y_train == a],
                                    y_train[y_train == a],
                                    n_samples=X_train[y_train == b].shape[0])
    X_train = np.concatenate((X_train[y_train == b], X_train_temp))
    y_train = np.concatenate((y_train[y_train == b], y_train_temp))
  return (X_train, X_test, y_train, y_test)


def rocauc_plot(model, model_name, y_test, X_test):

  try:
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
  except:
    auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
  plt.plot(fpr, tpr, label=model_name+" AUC = {:.5f}".format(auc))
  plt.title("Curva(s) ROC", fontdict={"fontsize": 21})
  plt.xlabel("False positive rate", fontdict={"fontsize": 13})
  plt.ylabel("True positive rate", fontdict={"fontsize": 13})
  plt.legend(loc="lower right")
  plt.plot([0, 1], [0, 1], "r--")

  def plot_roc_conf_matrix(y_test,X_test, model, model_name):

    try:
      y_pred=model.predict_classes(X_test)
    except:
     y_pred=model.predict(X_test)
  cm = metrics.confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(15,5))
  plt.subplot(1,2,1)
  sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
  plt.title(model_name+ " - Matriz de confusión", y=1.1, \
            fontdict={"fontsize": 21})
  plt.xlabel("Predicted", fontdict={"fontsize": 14})
  plt.ylabel("Actual", fontdict={"fontsize": 14})
 
  print(classification_report(y_test, y_pred))
  plt.subplot(1,2,2)

  rocauc_plot(model, model_name, y_test, X_test)

  
def apply_model_to_df(data, model, model_name):

  X_train, X_test, y_train, y_test=data
  model.fit(X_train, y_train)
  y_predi = model.predict(X_test)
  plot_roc_conf_matrix(y_test,X_test, model, model_name)
  return model

X,y =get_X_y(rango_mes,'cantidad')
X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.2)



model=make_pipeline(StandardScaler(),RandomForestRegressor())

model.fit(X_train,y_train)

y_predi=model.predict(X_test)
error=mean_squared_error(y_true=y_test,y_pred=y_predi)


test=pd.DataFrame([y_predi,y_test]).T.rename(columns={0:'y_predi',1:'y_test'})
test['y_predi']=test['y_predi'].apply(lambda x: round(x))


test['dif']=test['y_predi']-test['y_test']
test.sort_values(by='dif').head()

prediccion=test.dif.apply(lambda x: x==0).sum()/len(test)*100

prediccionReal=(100-prediccion)


st.write("LA PROBABILIDAD DE QUE HAYA HOMICIDIOS EN EL MUNICIPIO DE ",selecMunicipio,
   " UBICADO EN EL DEPARTAMENTO DE ",departamento, "EN EL MES ",selecmes," DEL AÑO ACTUAL ES DE ",
  prediccionReal,"%")