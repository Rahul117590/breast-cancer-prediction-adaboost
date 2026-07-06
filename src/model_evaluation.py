import os
import logging
import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

# create the file 
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# create the logger
logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

#create the streamHandler
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# create the filehandler
log_dir=os.path.join(log_dir,'model_evaluation.log')
file_handler=logging.FileHandler(log_dir)
file_handler.setLevel('DEBUG')

# set formatter
formatter=logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# add logger to handler
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# now load the data 
def load_model(model_path:str):
    ''' model is loading from the model_path'''
    try:
        model=joblib.load(model_path)
        logger.debug(f'the model is loading from the path of {model_path}')
        return model

    except FileNotFoundError as e:
        logger.error('the error due to model do not found ')
        raise
    except Exception as e:
        logger.error(f'Unexpected error found due to {e}')
        raise

def load_data(test_path:str):
    ''' load the test data'''
    try:
        test_data=pd.read_csv(test_path)
        logger.debug(f'the test_data is load from the path of {test_path}')

        X_test=test_data.drop(columns=['diagnosis'])
        y_test=test_data['diagnosis']

        logger.debug(f' the data is split in the form of {X_test.shape} and {y_test.shape}')
        return X_test,y_test

    except FileNotFoundError as e:
        logger.error(f'the data do not found ')
        raise

    except Exception as e:
        logger.error(f'Unexpected error found in {e}')
        raise

def evaulate_model(model,X_test,y_test):
    ''' model evaulation is started here '''
    try:
        y_pred=model.predict(X_test)
        logger.debug(f'the prediction of model ')

        acc=accuracy_score(y_test,y_pred)
        preci=precision_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred)
        rec=recall_score(y_test,y_pred)
        cm=confusion_matrix(y_test,y_pred)
        logger.info('the evaulation is completed')

        print(f'the accuracy of the model is :{acc:.2f} and {acc*100:.2f}')
        print(f'the precision of the model is :{preci:.4f}')
        print(f'the f1_score of the model is :{f1:.4f}')
        print(f'the recall_score of the model is : {rec:.4f}')
        print(f'confusion metrix :{cm}')
        logger.info('the metrics evaulation is complete')

        metrics={
            'accuracy':acc,
            'percision':preci,
            'f1_score':f1,
            'recall_score':rec,
            'confusion_metrics':cm.tolist()
        }

        return metrics
    except ValueError as e: # this error may be data_type be different
        logger.error(f'error may be generated due to data type is different {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error may be {e}')
        raise

def save_metrics(metrics:dict,metrics_path:str):
    ''' the metrics result file save in the json file '''
    try:
        os.makedirs(os.path.dirname(metrics_path),exist_ok=True)
        with open (metrics_path,'w') as f:
            json.dump(metrics,f,indent=4)#indent means that simple for format

        logger.info('the jsom file is save successfully')

    except Exception as e:
        logger.error(f'Unexpected error found due to {e}')
        raise

def main():
    ''' final evaluatin is start from here '''
    try:
        model_path=os.path.join('models','model.pkl')
        test_path=os.path.join('data','interim','test_scaled.csv')
        metrics_path=os.path.join('report','metrics.json')
        
        # load the model 
        model=load_model(model_path)
        # load the data 
        X_test,y_test=load_data(test_path)
        metrics=evaulate_model(model,X_test,y_test)
        save_metrics(metrics,metrics_path)

        logger.info('the model_evaluation is comleted')

    except FileNotFoundError as e:
        logger.error('the error happend due to file not found')
        raise

    except Exception as e:
        logger.error(f'Unexpected error found due to {e}')
        raise

if __name__=='__main__':
    main()








    