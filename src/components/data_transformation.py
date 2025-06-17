import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
        
    def get_data_transformation(self):

        try:
            
            numerical_features=['reading_score','writing_score']
            categorical_features=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            ##creating pipeline for numerical features

            num_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])

            #3creating pipeline for categorical data

            cat_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OneHotEncoder()),
                ('scaler',StandardScaler())
            ])

            

            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_features),
                ('cat_pipeline',cat_pipeline,categorical_features)
            ])

            return preprocessor
        except Exception as e:
            raise Exception("Encountering error in transformation pipeline")

    def initiate_data_transformation(self,train_path,test_path):
        try:
            ##loading of dataset
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            input_features_train=train_df.drop(columns=['math_score'])
            target_features_train=train_df['math_score']

            input_features_test=test_df.drop(columns=['math_score'])
            target_features_test=test_df['math_score']

            ## getting preprocessor pipeline

            preprocessor_obj=self.get_data_transformation()

            #3 applying transformation 

            input_feature_train_trans=preprocessor_obj.fit_transform(input_features_train)
            input_feature_test_trans=preprocessor_obj.fit_transform(input_features_test)

            ##after transformation of input features combine both input and target features


            train_trans=np.c_[input_feature_train_trans,np_array(target_features_train)] ## np.c_ for adding column wise and for changing 
            test_trans=np.c_[input_feature_test_trans,np_array(target_features_test)]

            ## save processor for future use 
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,obj=preprocessor_obj
            )

            return (
            train_trans,
            test_trans,
            self.transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise Exception("error in initiating data transformation")


