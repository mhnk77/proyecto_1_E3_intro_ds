"------------------------------------------------------------------------------"
#############
## Imports ##
#############

## Python libraries
import pandas as pd
#import re
import unicodedata


"------------------------------------------------------------------------------"
###############
## Functions ##
###############

def cla(col):
    return col.lower().replace('/','_').replace(' ','_').replace('ñ','n')
 
def clean_column(data):
    data.rename(columns={col: cla(col) for col in data.columns.values}, inplace=True)
    return data
 

def getVariables(data):
    # variables numéricas
    num_var = data.select_dtypes(include = 'number').columns.values
    # fechas 
    date_var = data.select_dtypes(include = 'datetime').columns.values
    # variables categóricas
    cat_var = data.select_dtypes(include = 'category').columns.values
    # strings 
    str_var = data.select_dtypes(include = 'object').columns.values
 
    varL = { 
       "num_var":(len(num_var), num_var),
       "date_var":(len(date_var), date_var),
       "cat_var": (len(cat_var), cat_var),
       "str_var":(len(str_var), str_var)
    }
  
    return varL

  
def changeType(data,list_Col,d_type):
    for name in list_Col:
        data[name] = data[name].astype(d_type)
    return data

   
   
def datetype_change(df,cols):
    for i in cols:
        df[i] = pd.to_datetime(df[i].str.strip(),format ='%d/%m/%Y')
    return
 
def checkDateForm(df,cols):
    for i in cols:
        che = df[df[i].str.match('^([0-2][0-9]|3[0-1])(\/|-)(0[1-9]|1[0-2])(\/|-)(\d{4})$')==False]
        s = pd.Series(che[i].value_counts().index).str.extract(r'(\d{2})$')[0].value_counts()
        print("\nCantidad de valores en "+i+" que no cumplen con el formato dd/mm/aaaa : {}".format(len(che)))
        print("Distribucion de valores:\n {}".format(s))
    return   

  
  
def checkTimeHour(df,cols):
    for i in cols:
        z = df[df[i].str.match('^([01]?[0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]$')==False]
        print("\nCantidad de valores en "+i+" que no cumplen con el formato hh/mm/ss : {}".format(len(z)))
        print("Distribucion de valores:\n {}".format(z[i].value_counts()))
    return
   
   

   

  
def convert_lower(data, vars_lower):
    """
     Converting observatios for selected columns into lowercase.
        args:
            data (dataframe): data that is being analyzed.
            vars_lower (list): list of the columns' names in the dataframe that will be changed to lowercase.
        returns:
            data(dataframe): dataframe that is being analyzed with the observations (of the selected columns) in lowercase.
    """
    for i in vars_lower:
        data[i]=data[i].str.lower()
    return data
   
   
   
def count_data(data):
    """
    Counting number of variables and obs in the data
        args:
            data (dataframe): data that is being analyzed
        returns:
            print : number of variables and obs in the data
    """
    listDF = []
    for name in data:
        #print(name)
        cT= data[name].value_counts()
        can_Cat = len(cT)

        top1 = cT.index[0]  if can_Cat >= 1 else 0
        fr1  = cT.values[0] if can_Cat >= 1 else 0
        top2 = cT.index[1] if can_Cat >= 2 else 0
        fr2  = cT.values[1] if can_Cat >= 1 else 0
        top3 = cT.index[2] if can_Cat >= 3 else 0
        fr3  = cT.values[2] if can_Cat >= 1 else 0
        
        moda = data[name].mode()
        modaVal = moda.values[0] if len(moda) == 1 else "valores unicos" 
        dpVar = { 
            "columnas":name,
            "tipo_dato":data[name].dtype,
            "cantidad":data[name].count(),
            "nulos":data[name].isnull().sum(),
            "unicos":data[name].nunique(),
            "moda":modaVal,
            "top1":top1,
            "f1":fr1,
            "top2":top2,
            "f2":fr2,
            "top3":top3,
            "f3":fr3}
        listDF.append(dpVar)
  
    dtyp = pd.DataFrame(listDF).set_index("columnas")
   
   
    nullSum = dtyp["nulos"].sum()
    prop  = round(100*(nullSum/data.shape[0]),2) 
    cdType = dtyp['tipo_dato'].value_counts()
 
 
    listAna = ["columnas","datos","total_nulos","%_nulos"]
    listInfo = [data.shape[1],data.shape[0],nullSum,"{}%".format(prop)]
    varDic = getVariables(data)
    
    for i in varDic:
        listAna.append(i)
        listInfo.append(varDic[i][0])
    
    infoD = { 
        "analisis":listAna,
        "informacion":listInfo
    }
 
    dfProp = pd.DataFrame(infoD).set_index("analisis").T
    #mostrar primer data frame
    print(display(dtyp))
    print("\n *********** Informacion global *************")
    print(display(dfProp))
    
    return

   
   
 
## Counting number of unique observations for all variables
def count_unique_obs(data):
    """
    Counting number of unique observations for all variables and show the types of variables after transformation
        args:
        data (dataframe): data that is being analyzed
        returns:
        (series): number of unique observations for all variables
    """
  
  
    return data.nunique() 
 
 

def proporcion(listaVar,n):
    """
    Calculate the data proportion of categorical variables.
     args:
         listaVar (Serie): Serie with unique values of categorical variables to get use value_counts() into a Serie 
         n (int): value of total observation of data set.
     returns:
         newList(list): List with name, count and proportion of each category
    """
    newList = []
    for lis in listaVar.iteritems():
          newList.append([lis[0],lis[1],"{}%".format(round(100*(lis[1]/n),1))])
      
    return newList

def data_profiling_categ(data, cat_vars):
    """
    Create the data profiling of categorical variables.
        args:
            data (Data Frame): data set into Dataframe.
            cat_vars (list): list with categorical variables names.
        returns:
           df_catVar: Dataframes with info.
    """
    listCat = []
    for val in cat_vars:
        catego  = data[val].value_counts()
        totalOb = len(data[val])
        can_Cat = len(catego)
        cat = data[val].unique()
        moda    = data[val].mode().values[0]
        valFal  = data[val].isnull().sum()
        top1 = catego.index[0]  if can_Cat >= 1 else 0
        fr1  = catego.values[0] if can_Cat >= 1 else 0
        top2 = catego.index[1] if can_Cat >= 2 else 0
        fr2  = catego.values[1] if can_Cat >= 1 else 0
        top3 = catego.index[2] if can_Cat >= 3 else 0
        fr3  = catego.values[2] if can_Cat >= 1 else 0
        elemVarCat = { 
            "metrica":val,
            "registros":totalOb,
            "nulos":valFal,
            "Num_categorias":can_Cat,
            "categorias":cat,
            "moda":moda,
            "top1":top1,
            "f1":fr1,
            "top2":top2,
            "f2":fr2,
            "top3":top3,
            "f3":fr3}
        listCat.append(elemVarCat)
    #primerdataframe   
    df_catVar = pd.DataFrame(listCat).set_index("metrica").T
    
    return df_catVar


   
   

 
 
def data_profiling_num(data,num_vars):
    """
    Create the data profiling of numeric variables
        args:
            data (dataframe): data that is being analyzed
            num_vars (list): list with numeric variables names
        returns:
            df_Num : Dataframes with info.
    """
    listNum = []
    for name in num_vars:
        #print(name)
        cT= data[name].value_counts()
        can_Cat = len(cT)

        top1 = cT.index[0]  if can_Cat >= 1 else 0
        fr1  = cT.values[0] if can_Cat >= 1 else 0
        top2 = cT.index[1] if can_Cat >= 2 else 0
        fr2  = cT.values[1] if can_Cat >= 1 else 0
        top3 = cT.index[2] if can_Cat >= 3 else 0
        fr3  = cT.values[2] if can_Cat >= 1 else 0
        
        moda = data[name].mode()
        modaVal = moda.values[0] if len(moda) == 1 else "valores unicos" 
        dpVar = { 
            "metricas":name,
            "registros":data[name].count(),
            "nulos":data[name].isnull().sum(),
            "unicos":data[name].nunique(),
            "moda":modaVal,
            "min": data[name].min(),
            "max": data[name].max(),           
            "mean": data[name].mean(),
            "stdv": data[name].std(),
            "25%": data[name].quantile(.25),
            "median": data[name].median(),
            "75%": data[name].quantile(.75),
            "kurtosis": data[name].kurt(),
            "skewness": data[name].skew(),
            "top1":top1,
            "f1":fr1,
            "top2":top2,
            "f2":fr2,
            "top3":top3,
            "f3":fr3}
        listNum.append(dpVar)
        
    df_Num = pd.DataFrame(listNum).set_index("metricas").T
    return df_Num

   
   
def data_profiling_date(data, date_vars):
    """
    Create the data profiling of date variables.
        args:
            data (Data Frame): data set into Dataframe.
            date_vars (list): list with date variables names.
        returns:
           df_dateVar: Dataframes with info.
    """
    listDate = []
    for name in date_vars:
        
        cT= data[name].value_counts()
        can_Cat = len(cT)
        
        moda    = data[name].mode()
        top1 = cT.index[0]  if can_Cat >= 1 else 0
        fr1  = cT.values[0] if can_Cat >= 1 else 0
        top2 = cT.index[1] if can_Cat >= 2 else 0
        fr2  = cT.values[1] if can_Cat >= 1 else 0
        top3 = cT.index[2] if can_Cat >= 3 else 0
        fr3  = cT.values[2] if can_Cat >= 1 else 0
        varDate = { 
            "metricas":name,
            "tipo dato": type(data[name].values[0]),
            "registros":data[name].count(),
            "nulos":data[name].isnull().sum(),
            "unicos":data[name].nunique(),
            "moda":moda,
            "fecha inicio":data[name].min(),
            "fecha fin":data[name].max(),
            "top1":top1,
            "f1":fr1,
            "top2":top2,
            "f2":fr2,
            "top3":top3,
            "f3":fr3}
        listDate.append(varDate)
    #primerdataframe   
    df_dateVar = pd.DataFrame(listDate).set_index("metricas").T
    
    return df_dateVar
#codigo para extraer la letra de codigo de cierre
#data_clean["codigo_cierre"].str.extract('.*\(([A-Z])\).*')[0].value_counts()

##data_clean[data_clean.folio=="C5/200112/09235"]