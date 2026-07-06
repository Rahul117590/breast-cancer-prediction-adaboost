import os
import logging
import joblib
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd



# create the file 
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# set the logger
logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

# set streamHandler
console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

# file_handler
log_dir=os.path.join(log_dir,'featuring_engineering.log')
file_handler=logging.FileHandler(log_dir)
file_handler.setLevel("DEBUG")

# setformatter
formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# add the handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)



# now the time of data load
def load_data(train_path:str,test_path:str):
    '''loading the data is start from here '''
    try:
        train_df=pd.read_csv(train_path)
        test_df=pd.read_csv(test_path)
        logger.info('the data is sucessfully is loaded')

        X_train=train_df.drop(columns=['diagnosis'])
        y_train=train_df['diagnosis']
        X_test=test_df.drop(columns=['diagnosis'])
        y_test=test_df['diagnosis']
        logger.info('the data is successfully split ')
        return X_train,X_test,y_train,y_test

    except FileNotFoundError as e:
        logger.error(f'the file may be not found')
        raise
    except Exception as e:
        logger.error(f'Unexcpeted error occur due to {e}')
        raise

def model_training(X_train,y_train):
    ''' model training is started from here'''
    try:
        model=AdaBoostClassifier(
            n_estimators=100,
            learning_rate=1.0,
            random_state=42
        )
        # train the model
        model.fit(X_train,y_train)
        logger.debug(f'the data is trained  with {model}')

        
        logging.info('the data training is sucessfully completed ')
        return model

    except ValueError as e:
        logger.error('may be the data size of X-train and y_train is different')
        raise
    except Exception as e:
        logger.error(f'Unexpected error may be happend with {e}')

def save_model(model,model_path:str):
    ''' here model is going to be saved '''
    try:
        joblib.dump(model,model_path)
        logger.debug(f'the model is saved in the {model_path}')

    except TypeError as e: # this error occure when the path is not found
        logger.error('may be path error due to that file not saved')
        raise
    except Exception as e:
        logger.error(f'Unexpected error may be occur due to {e}')
        raise

def main():
    ''' feature engineering started here '''
    # find the path 
    try:
        train_path=os.path.join('data','interim','train_scaled.csv')
        test_path=os.path.join('data','interim','test_scaled.csv')
        # make the model_path
        model_path=os.path.join('models','model.pkl')
        os.makedirs(os.path.dirname(model_path),exist_ok=True)


        X_train,X_test,y_train,y_test=load_data(train_path,test_path)
        model=model_training(X_train,y_train)
        save_model(model,model_path)
        logger.debug(f'the model is save ini the {model_path}')
    except FileExistsError as e:
        logger.error(f'the file is not found here {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error happend due to {e}')
        raise
    
if __name__=='__main__':
    main()



    

