## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings
import pickle
warnings.filterwarnings("ignore")

# 1. TimeGAN model
from timegan import timegan
# 2. Data loading
from data_loading import real_data_loading, sine_data_generation
# 3. Metrics
from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics import predictive_score_metrics
from metrics.visualization_metrics import visualization

## Data loading
data_name = 'head'
seq_len = 1000

if data_name in ['stock', 'energy','head']:
  ori_data = real_data_loading(data_name, seq_len)
elif data_name == 'sine':
  # Set number of samples and its dimensions
  no, dim = 10000, 5
  ori_data = sine_data_generation(no, seq_len, dim)
    
print("LEN", len(ori_data), ori_data[0].shape)
print(data_name + ' dataset is ready.')

## Newtork parameters
parameters = dict()

parameters['module'] = 'gru' 
parameters['hidden_dim'] = 24
parameters['num_layer'] = 3
parameters['iterations'] = 10000
parameters['batch_size'] = 128

# Run TimeGAN
generated_data = timegan(ori_data, parameters)   
print(len(generated_data), generated_data[0].shape)
print('Finish Synthetic Data Generation')

pickle.dump(generated_data, open("gen.out", "wb"))

#metric_iteration = 5

#discriminative_score = list()
#for _ in range(metric_iteration):
#  temp_disc = discriminative_score_metrics(ori_data, generated_data)
#  discriminative_score.append(temp_disc)

#print('Discriminative score: ' + str(np.round(np.mean(discriminative_score), 4)))

#predictive_score = list()
#for tt in range(metric_iteration):
#  temp_pred = predictive_score_metrics(ori_data, generated_data)
#  predictive_score.append(temp_pred)   
    
#print('Predictive score: ' + str(np.round(np.mean(predictive_score), 4)))

visualization(ori_data, generated_data, 'pca')
visualization(ori_data, generated_data, 'tsne')
