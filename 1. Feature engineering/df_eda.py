import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings 
warnings.filterwarnings(action='ignore')

class df_eda:
    
    pd.options.display.float_format = '{:.2f}'.format
    
    print("df_describe(self, df, ID, t_object=True)")
    print("df_graph(self, df, t_object = True)")
    
    
    def df_describe(self, df, ID, t_object=True) :
    
    # df : 테이블명 
    # ID : Unique count 기준이 되는 ID 
    
        if t_object : 
        
            key_list = list()
            value_list = list() 
            cust_max_list = list()
            cust_max_list2 = list()
            cust_max_value = list()
            cust_max_value2 = list() 
            for key, value in dict(df.dtypes == object).items() : 
                if value : 
                    key_list.append(key) 
                    value_list.append(df[key].nunique())
                    cust_max_list.append(df.groupby(key)[ID].nunique().sort_values(ascending =False).keys()[0])
                    cust_max_value.append(df[key].value_counts(normalize=True).sort_values(ascending =False)[0])
                    cust_max_list2.append(df.groupby(key)[ID].nunique().sort_values(ascending =False).keys()[1])
                    cust_max_value2.append(df[key].value_counts(normalize=True).sort_values(ascending =False)[1])
            result = pd.DataFrame({'column' : key_list
                                 ,'label_count' : value_list 
                                 ,'Top1_label':cust_max_list   
                                 ,'Top1_percent':cust_max_value   
                                 ,'Top2_label':cust_max_list2  
                                 ,'Top2_percent':cust_max_value2  
                                 })
            
        else : 
            num_df = df[list(dict(df.dtypes != object).keys())]
            result = num_df.describe().T.reset_index()
            
            
        return result
    
    def df_graph(self, df, t_object = True) : 
        
        summ_df = pd.DataFrame()
        
        def count_outliers(df, col):
            mean_d = np.mean(df[col])
            std_d = np.std(df[col])

            scaled = (df[col]-mean_d)/std_d
            outliers = abs(scaled) > 3
            if len(outliers.value_counts()) > 1:
                return outliers.value_counts()[1]
            else:
                return 0  
        
        for col, value in dict(df.dtypes == object).items() : 
            if value & t_object : 
                count_u = df[col].nunique()
                nulls = df[col].isnull().sum()

                 ### Percent share df
                share_df = pd.DataFrame(df[col].value_counts()).reset_index().rename(columns={'index':'class_name',col:'counts'})
                share_df['percent_share'] = share_df['counts']/sum(share_df['counts'])
                share_df = share_df.sort_values(by='percent_share', ascending=False)
                
                if (count_u > 3 and count_u < 10):
                    fig, ax  = plt.subplots()
                    fig.suptitle(col + ' Distribution', color = 'red')
                    explode = list((np.array(list(df[col].dropna().value_counts()))/sum(list(df[col].dropna().value_counts())))[::-1])
                    labels = list(df[col].dropna().unique())
                    sizes = df[col].value_counts()
                    #ax.pie(sizes, explode=explode, colors=bo, startangle=60, labels=labels,autopct='%1.0f%%', pctdistance=0.9)
                    cmap = plt.get_cmap('Pastel1')
                    colors = [cmap(i) for i in np.linspace(0, 1, 8)]

                    ax.pie(sizes,  explode=explode, startangle=60, labels=labels,autopct='%1.0f%%', pctdistance=0.9, colors=colors)
                    ax.add_artist(plt.Circle((0,0),0.2,fc='white'))
                    print('\n')
                    print("#########"+col+"##########")
                    print("Unique value : {}, Missing value : {}".format(count_u, nulls, nulls/len(df)))
                    plt.show()

                else :
                    plt.figure()
                    plt.title(col + ' Distribution')
                    sns.countplot(y =col, data = df, palette="Set3" )
                    print("#########"+col+"##########")
                    print("Unique value : {}, Missing value : {}".format(count_u, nulls, nulls/len(df)))
                    plt.show()
                    
            elif (value == False) & (t_object == False): 
                count_df = df[col].isnull().sum()
                df_k = stats.kurtosis(df[col].dropna(), bias=False)    
                df_s = stats.skew(df[col].dropna(), bias=False)
                df_o = count_outliers(df, col)
                
                summ_df[col] = [count_df, (count_df/len(df))*100, df_s, df_k, df_o, (df_o/len(df))*100]
    
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))
    
                plot10 = sns.distplot(df[df['Attrition_Flag']=="Existing Customer"][col],ax=ax1, label='Existing Customer')
                sns.distplot(df[df['Attrition_Flag']=="Attrited Customer"][col],ax=ax1,color='red', label='Attrited Customer')
                plot10.axes.legend()
                ax1.set_title('Distribution of {name}'.format(name=col))

                sns.boxplot(x='Attrition_Flag',y=col,data=df,ax=ax2, palette="Set3" )
                #plt.xticks(ticks=[0,1],labels=['Non-Diabetes','Diabetes'])
                ax2.set_xlabel('Category') 
                ax2.set_title('Boxplot of {name}'.format(name=col))
            
                fig.show()
        if t_object == False :         
            index = ['missing_count', 'missing_percent', 'skewness', 'kurtosis', 'outlier_count', 'outlier_percent']
            summ_df.index = index
    
            print("Summary Data :")
            display(summ_df.style.background_gradient(cmap='Reds', axis=1).format('{:.2f}'))