""" 
model.py

The script used to create and train the model.
"""

import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, args):
        self.data = args.data
        self.train_test_split = args.train_test_split
        self.dropout = args.dropout
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.samples_per_epoch = args.samples_per_epoch
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
    
    def load_data():
        """
        Load training data and split it into training and validation set
        """
        data_df = pd.read_csv(os.path.join(args.data, 'driving_log.csv'))

        X = data_df[['center', 'left', 'right']].values
        y = data_df['steering'].values

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    # def train():
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavioral Cloning Model')
    parser.add_argument(
        'data',
        type=str,
        help='data directory',
        default='data'
    )
    parser.add_argument(
        'train_test_split',
        type=float,
        help='train test split fraction',            
        default=0.2
    )
    parser.add_argument(
        'dropout',
        type=float, 
        help='drop out probability',          
        default=0.5
    )
    parser.add_argument(
        'epoch',
        type=int,
        help='number of epochs',
        default=10
    )
    parser.add_argument(
        'samples_per_epoch',
        type=int,
        help='samples per epoch',
        default=20000
        )
    parser.add_argument(
        'batch_size',
        type=int,
        help='batch size',
        default=40
    )
    parser.add_argument(
        'learning_rate',
        type=float,
        help='learning rate',
        default=1.0e-4
    )
    args = parser.parse_args()

    model = Model(args)
    model.load_data()
