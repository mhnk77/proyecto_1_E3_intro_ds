#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:25:01 2020

@author: mario
"""

import pandas as pd
import pickle
import numpy as np



data = pd.read_csv('data/incidentes-viales-c5.csv')
data.head()

data_clean = data.copy(deep = True) #Copy deep para que los cambios hechos a data_clean no se vean reflejados en data

#Existen renglones en los que el formato de fecha es "dd/mm/aa" en vez de "dd/mm/aaaa", para ambias variables de fecha
#Corregimos esto mediante:

data_clean['fecha_creacion'] = data['fecha_creacion'].replace("/19$", "/2019", regex = True, inplace = False) 
data_clean['fecha_creacion'] = data_clean['fecha_creacion'].replace("/18$", "/2018", regex = True, inplace = False) 
data_clean['fecha_cierre'] = data['fecha_cierre'].replace("/19$", "/2019", regex = True, inplace = False) 
data_clean['fecha_cierre'] = data_clean['fecha_cierre'].replace("/18$", "/2018", regex = True, inplace = False) 

#Convertir en fecha las variables correspondientes
data_clean['fecha_creacion'] = pd.to_datetime(data_clean['fecha_creacion'], format = '%d/%m/%Y')
data_clean['fecha_cierre'] = pd.to_datetime(data_clean['fecha_cierre'], format = '%d/%m/%Y')

#Transformar a minusculas las delegaciones
data_clean['delegacion_inicio'] = data.delegacion_inicio.str.lower()
data_clean['delegacion_cierre'] = data.delegacion_cierre.str.lower()

## Modificar incidente c4
#Quitar mayúsculas
data_clean['incidente_c4'] = data_clean['incidente_c4'].str.lower()
#Quitar acentos
data_clean['incidente_c4'] = data_clean['incidente_c4'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
#Quitar " /"
data_clean['incidente_c4'] = data_clean['incidente_c4'].str.replace(' /','', regex =  True)
#Cambiar espacios y "-" por "_"
data_clean['incidente_c4'] = data_clean['incidente_c4'].str.replace('\-|\s','_', regex =  True)

#Conservar únicamente la letra del codigo_cierre
data_clean['codigo_cierre'] = data_clean['codigo_cierre'].str.extract(r'(\(.\))')
data_clean['codigo_cierre'] = data_clean['codigo_cierre'].str.replace('\(|\)', '', regex = True)

#Eliminar la columna "geopoint" pues ya tenemos long. y lat.
data_clean.drop(columns = ['geopoint'], inplace = True)


#Crear variables dttm
data_clean['dttm_creacion'] = pd.to_datetime(data_clean.fecha_creacion.astype(str).str.cat(data_clean.hora_creacion, sep = ' '), errors= 'coerce')
data_clean['dttm_cierre'] = pd.to_datetime(data_clean.fecha_cierre.astype(str).str.cat(data_clean.hora_cierre, sep = ' '), errors= 'coerce')

#Corregir las fechas donde el formato de la hora es decimal
data_clean.loc[data_clean.dttm_creacion.isna(), 'dttm_creacion']= (data_clean[data_clean.dttm_creacion.isna()]['fecha_creacion'] + pd.to_timedelta(data_clean[data_clean.dttm_creacion.isna()].hora_creacion.astype(float), 'days'))
data_clean.loc[data_clean.dttm_cierre.isna(), 'dttm_cierre']= (data_clean[data_clean.dttm_cierre.isna()]['fecha_cierre'] + pd.to_timedelta(data_clean[data_clean.dttm_cierre.isna()].hora_cierre.astype(float), 'days'))

#Quitar variables relacionadas con fecha
data_clean.drop(columns = ['fecha_creacion', 'hora_creacion', 'dia_semana','fecha_cierre', 'año_cierre', 'mes_cierre', 'hora_cierre', 'mes'], inplace = True)

#Quitamos los renglones con faltantes en geopoint, delegacion, y latitud o longitud
data_clean.dropna(inplace = True)

#Crear variables derivadadas de dttm
data_clean['dow_creacion'] = data_clean['dttm_creacion'].dt.dayofweek
data_clean['hora_creacion'] = data_clean['dttm_creacion'].dt.hour
data_clean['mes_creacion'] = data_clean['dttm_creacion'].dt.month
data_clean['año_creacion'] = data_clean['dttm_creacion'].dt.year
data_clean['fecha_creacion'] = data_clean['dttm_creacion'].dt.floor('D')
data_clean['dow_cierre'] = data_clean['dttm_cierre'].dt.dayofweek
data_clean['hora_cierre'] = data_clean['dttm_cierre'].dt.hour
data_clean['mes_cierre'] = data_clean['dttm_cierre'].dt.month
data_clean['año_cierre'] = data_clean['dttm_cierre'].dt.year
data_clean['fecha_cierre'] = data_clean['dttm_cierre'].dt.floor('D')

#Crear nuevo geopoint
data_clean['geopoint'] = data_clean['longitud'].round(5).astype('string')+','+data_clean['latitud'].round(5).astype('string')

#Cambiar a minusculas y quitar acentos
data_clean['clas_con_f_alarma'] = data_clean['clas_con_f_alarma'].str.lower().replace(" ", "_", regex = True, inplace=False)
data_clean['tipo_entrada'] = data_clean['tipo_entrada'].str.lower().replace(" ", "_", regex = True, inplace=False).str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

#creación de variable de respuesta 'label'
data_clean['label'] = np.where((data_clean['codigo_cierre'] == 'N') | (data_clean['codigo_cierre'] == 'F') ,1,0 )

#Quitamos todo lo que sea de cierre ya que no contaremos con esas variables al momento de predecir
data_rf = data_clean.loc[:,~data_clean.columns.str.contains('_cierre')]

#Quitamos el resto de las variables que no serán útiles
data_rf = data_rf.drop(columns = ['folio', 'clas_con_f_alarma', 'latitud', 'longitud', 'dttm_creacion', 'fecha_creacion', 'año_creacion'])

## Feature engineering

incidentes_top = data_rf.incidente_c4.value_counts().head(9).reset_index(name = "n")['index'].values

data_rf['incidente'] = np.where((data_rf['incidente_c4'].isin(incidentes_top)) ,data_rf['incidente_c4'],"otros" )

geopoints_top = data_rf.geopoint.value_counts().reset_index(name = "n").query('n >= 300')['index'].values

data_rf['tipo_geopoint'] = np.where((data_rf['geopoint'].isin(geopoints_top)) , "frecuente","aislado" )

tipo_entrada_top = data_rf.tipo_entrada.value_counts().head(4).reset_index(name = "n")['index'].values

data_rf['tipo_entrada_mod'] = np.where((data_rf['tipo_entrada'].isin(tipo_entrada_top)) ,data_rf['tipo_entrada'],"otros" )

data_rf['delegacion_inicio'] = data_rf['delegacion_inicio'].str.replace(" ", "_", regex = True)

data_rf = data_rf.drop(columns = ['geopoint','incidente_c4','tipo_entrada'])

#Encoding
#otros estamos

data_encoded = pd.get_dummies(data_rf, columns = ['delegacion_inicio', 'incidente', 'tipo_geopoint', 'tipo_entrada_mod'])










