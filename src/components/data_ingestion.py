import os
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')

class Dataingestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df=pd.read_csv("notebook\data\stud.csv")

            ##To create artifacts folder ---could  have used any  train test or raw 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)  ## whenever we try to create directories we need to give folder path not the file name

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            ##train test split
            train_set,test_set=train_test_split(df, test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)## saving train set into train data path i.e artifact folder

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)## saving test set into test data path i.e artifact folder

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
                
            )

        except Exception as e:
            raise Exception('error loading data ')
        
if __name__=="__main__":
    obj=Dataingestion()
    obj.initiate_data_ingestion()