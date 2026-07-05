import pandas as pd
import logging 
import os
from sklearn.model_selection import train_test_split

# make file 
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# set logging
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# console_handler--> it means that print the error on the screen
console_handler=logging.StreamHandler()
console_handler.SetLevel('DEBUG')

# set file Handler --> it mean that print error in the log file 
log_dir=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_dir)
file_handler.SetLevel("DEBUG")

# set formatter
formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

#adder
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def data_load(data_path:str) ->pd.DataFrame:
    ''' this is going to load the data form the github data '''
    try:
        df=pd.read_csv(data_path)
        df=pd.DataFrame(df)
        logger.debug(f'the data is logging from the {data_path}')
        return df
    except pd.error.ParseError as e:
        logger.error(f'Failed the loading the csv_file')
        raise
    except Exception as e:
        logger.error('Unexpected error happend during the loading the file from the csv folder')
        raise

def drop_unwanted_columne(df:pd.DateFrame) -> pd.DataFrame:
    ''' this is going to drop the unwanted columne of the data '''
    try:
        df=df.drop(columnes=['id'],inplace=True)
        logger.debug('the unwanted columne are drop sucessfully{df}')
        return df
    except Exception as e:
        logger.error('Unexcepted error happend during the data logging in form of df')
    
def split_data(df:pd.DataFrame,test_size:float)->tuple:
    ''' this is divide the data in the form of train and test form '''
    try:
        # divide the data using skcit 
        train_df,test_df=train_test_split(df,test_size=test_size,random_state=42)
        logger.info(f'data is successfully splited in {train_df} and {test_df}')
        return train_df,test_df
    except ValueError as e: # chances of test size error like 1.5
        logger.error(f' Value error may be in the train_test_split{e}')
        raise
    except Exception as e:
        logger.error("Unexpected error may be occur during the data split ")

def save(train_df:pd.DataFrame,test_df:pd.DataFrame,output_path:str) ->None:
    ''' this is save the data in from of train_df and test_df '''
    try:
        raw_data_path=os.path.join(output_path,'raw')
        os.makedir(raw_data_path,exist_ok=True)
        train_df.to_csv(os.path.join(raw_data_path,'train.csv'))
        test_df.to_csv(os.path.join(raw_data_path,'test.csv'))
        logger.info(f'the data in save in the file of {train_df} and {test_df}')
        
    except FileNotFoundError as e: # may be folder is missing during save the folder
        logger.error(f'Directory not found error {e}')
        raise
    except Exception as e:
        logger.error(f"Unexpected error found during save the folder {e}")
        raise

def main():
    ''' the data_ingestion process is start'''
    try:
        data_path=
        test_size=0.2
        df=data_load(data_path=data_path)
        df=drop_unwanted_columne(df)
        train_df,test_df=split_data(df,test_size)
        save(train_df,test_df,output_path=)
    
    except Exception as e:
        logger.error('un')
