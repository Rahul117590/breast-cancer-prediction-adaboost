import os
import pandas as pd
import logging
import joblib
from sklearn.preprocessing import StandardScaler

# create log file 
log_dir='log'
os.makedirs(log_dir,exist_ok=True)

#create the logger 
logger=logging.getLogger('feature_engineering.log')
logger.setLevel('DEBUG')

# create sreamhandler
console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

#file_handler
log_dir=os.path.join(log_dir,'feature_engineering.log') #connect the file with the log file
file_handler=logging.FileHandler(log_dir)
file_handler.setLevel('DEBUG')

#setformatter
formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s') #ceate the type of formatter
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#add logger with handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(train_path:str,test_path:str):
    ''' data loading process is strt from here '''
    try:
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)
        logger.info('the data loading is complete')
        return train_df,test_df


    except FileNotFoundError as e:
        logger.error(f'Data laoding is failed due to {e}')
        raise
        
    except Exception as e:
        logger.error(f'Unexected error due to {e}')
        raise

def scale_feature(train_df:str,test_df:str):
    ''' the data need to be standardized form it means 0 to 1 range'''
    try:
        #first we need to split the data 
        X_train=train_df.drop(columns=['diagnosis'])
        y_train=train_df['diagnosis']

        X_test=test_df.drop(columns=['diagnosis'])
        y_test=test_df['diagnosis']
        logger.info(f'the data is sucessfully splited')

        # now we apply the feature engineering 
        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        X_test_scaled=scaler.transform(X_test)
        logger.info('the data is successfully scaled')

        # tranfrom data is np.array form so we need to convert them info DataFrame
        X_train_scaled=pd.DataFrame(X_train_scaled,columns=X_train.columns)
        #columns=X_train.columns purpose that beacuse X_train_scaled output is in numpy that why
        X_test_scaled=pd.DataFrame(X_test_scaled,columns=X_test.columns)
        logger.info('the data is convert info DataFrame')

        return X_train_scaled,X_test_scaled,y_train,y_test,scaler
    
    except ValueError as e: # this error rasied dur to data could not convert string to float
        logger.error('the data could convert string to float')
        raise
    except Exception as e:
        logger.error(f'Unexcepected error may be occur due to{e}')
        raise

def save_scaler(scaler,scaler_path:str):
    ''' this function store the value of std,mean'''
    try:
       
        joblib.dump(scaler,scaler_path)
        logger.info('the scaler is successfully saved ')

    except FileNotFoundError as e:# it means that file not exist in that datapath
        logger.error(f'the error raised due to file not exist')
        raise
    except Exception as e:
        logger.error(f'Unexpected error found due to {e}')
        raise
def save_data(X_train_scaled:pd.DataFrame,X_test_scaled:pd.DataFrame,y_train,y_test,output_path):
    ''' this is save the data for furtur use '''
    try:
        X_train_scaled['diagnosis']=y_train
        X_test_scaled['diagnosis']=y_test

        #now we have to make the path of where we wanana store the data
        interim_data_path=os.path.join(output_path,'interim')# this the path where data is stored
        os.makedirs(interim_data_path,exist_ok=True)# this is the folder where data saved data/interim/imterim_data
        
        # now we have the data in this folder
        X_train_scaled.to_csv(os.path.join(interim_data_path,'train_scaled.csv'),index=False) #now 
        X_test_scaled.to_csv(os.path.join(interim_data_path,'test_scaled.csv'),index=False)
        logger.info(f'the data is saved sucessfully in the data_path')
    
    except FileNotFoundError as e:
        logger.error(f'the error due to file not found in the path')
        raise
    except Exception as e:
        logger.error(f'Unexpected error found due to {e}')
        raise

def main():
    ''' now main process of feature_engineering started form here '''
    try:
        train_path=os.path.join('data','processed','train_processed.csv')
        test_path=os.path.join('data','processed','test_processed.csv')
        # we need to make the scaler path for save-scaler
        scaler_path=os.path.join('models','scaler.pkl')
        os.makedirs(os.path.dirname(scaler_path),exist_ok=True)

        #load_data
        train_df,test_df=load_data(train_path,test_path)
        X_train_scaled,X_test_scaled,y_train,y_test,scaler=scale_feature(train_df,test_df)
        
        #save the sacler data 
        save_scaler(scaler,scaler_path)
        #now we save the data in the file 
        save_data(X_train_scaled,X_test_scaled,y_train,y_test,output_path='data')
        logger.debug(f'the data is store in the path of {scaler_path} ')

    except Exception as e:
        logger.error(f'Unexpected error found due to {e}')
        raise
        


if __name__=='__main__':
    main()

















