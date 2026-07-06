import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# create file
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# create logger
logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# create console_handler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# create file_handler
log_dir=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_dir)
file_handler.setLevel('DEBUG')

#formatter
formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#add loggerto handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_data(train_path:str,test_path:str):
    '''loading the data form the previous file '''
    try:
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)
        logger.info(f'the data is successfully loaded with the shape of {train_df.shape}and {test_df.shape}')
        return train_df,test_df
    
    except FileNotFoundError as e: # this is occur when file not loaded 
        logger.error(f'the error causesed by file not found error {e}')
        raise
    except  Exception as e:
        logger.error(f'Unexpected error caused due to data not loaded {e}')
        raise

def data_preprocessing(train_df:pd.DataFrame,test_df:pd.DataFrame)->pd.DataFrame:
    ''' remove duplicated and missing value'''
    try:
        #remvoe the duplicates
        train_df=train_df.drop_duplicates()
        test_df=test_df.drop_duplicates()
        logger.info('duplicates are successfully removed')
        # missing value remove 
        train_df=train_df.fillna(train_df.median(numeric_only=True))
        test_df=test_df.fillna(test_df.median(numeric_only=True))
        logger.info('missing value is removed successfully')
        return train_df,test_df

    except ValueError as e: # this value is occur when data contain the NAN value
        logger.error(f'file not loaded due to {e}') 
        raise

    except Exception as e: 
        logger.info('Unexcepted error happend due to {e}')
        raise
def data_encode(train_df:pd.DataFrame,test_df:pd.DataFrame):
    ''' data split into target columne'''
    try:
        encoder=LabelEncoder()
        train_df['diagnosis']=encoder.fit_transform(train_df['diagnosis'])
        test_df['diagnosis']=encoder.transform(test_df['diagnosis'])
        X_train=train_df.drop(columns=['diagnosis'])
        y_train=train_df['diagnosis']

        X_test=test_df.drop(columns=['diagnosis'])
        y_test=test_df['diagnosis']

        logger.info('the data is sucessfully encoded')
        return X_train,X_test,y_train,y_test
        
    except ValueError as e: # this is error caused by the data do not loading successfully 
        logger.error(f'the data encoding is failed {e}')
    except Exception as e:
        logger.error(f'Unexpected error caused due to {e}')

def save(X_train:pd.DataFrame,X_test:pd.DataFrame,y_train,y_test,output_path:str):
    ''' the data is storing processing is start'''
    try:
        #we need to convert into two for next requirement so 
        X_train['diagnosis']=y_train# y_train # it means that make a columne inside X_train that store all the value of 
        X_test['diagnosis']=y_test# y_train and the columne name  is diagonis
        processed_data_path=os.path.join(output_path,'processed')
        os.makedirs(processed_data_path,exist_ok=True)
        X_train.to_csv(os.path.join(processed_data_path,'train_processed.csv'),index=False)
        X_test.to_csv(os.path.join(processed_data_path,'test_processed.csv'),index=False)
        logger.info('the data is succesfully saved in the csv file ')
    

    except FileNotFoundError as e:
        logger.error(f'the file is not found in the form of {e}')
        raise

    except Exception as e:
        logger.error(f'Unexpected error found due to {e}')
        raise

def main():
    ''' data preprocesing is started '''
    try:
        train_path=os.path.join('data','raw','train.csv')
        test_path=os.path.join('data','raw','test.csv')
        #data is loading now
        train_df,test_df=load_data(train_path,test_path)
        # data is processing now
        train_df,test_df=data_preprocessing(train_df,test_df)
        # data encoding now
        X_train,X_test,y_train,y_test=data_encode(train_df,test_df)
        
        
        # now the data need to be saved 
        save(X_train,X_test,y_train,y_test,output_path='data')
        logger.debug(f'the data is store into the the {'output_path'}')

    except FileExistsError as e:# it means that error occur when file location already exist at that specific loaction
        logger.info(f'the error is come due  to {e}')
        raise
    except Exception as e:
        logger.info(f'Unexcepted error generted due to {e}')


if __name__=='__main__':
    main()


        
        



                
    
        






