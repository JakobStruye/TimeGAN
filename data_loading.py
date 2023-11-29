"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

data_loading.py

(0) MinMaxScaler: Min Max normalizer
(1) sine_data_generation: Generate sine dataset
(2) real_data_loading: Load and preprocess real data
  - stock_data: https://finance.yahoo.com/quote/GOOG/history?p=GOOG
  - energy_data: http://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
"""

## Necessary Packages
import numpy as np
import math
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, StandardScaler
import scipy.stats as scs
import os

def MinMaxScaler(data):
  """Min Max normalizer.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
  """
  numerator = data - np.min(data, 0)
  denominator = np.max(data, 0) - np.min(data, 0)
  norm_data = numerator / (denominator + 1e-7)
  return norm_data


def sine_data_generation (no, seq_len, dim):
  """Sine data generation.
  
  Args:
    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions
    
  Returns:
    - data: generated data
  """  
  # Initialize the output
  data = list()

  # Generate sine data
  for i in range(no):      
    # Initialize each time-series
    temp = list()
    # For each feature
    for k in range(dim):
      # Randomly drawn frequency and phase
      freq = np.random.uniform(0, 0.1)            
      phase = np.random.uniform(0, 0.1)
          
      # Generate sine signal based on the drawn frequency and phase
      temp_data = [np.sin(freq * j + phase) for j in range(seq_len)] 
      temp.append(temp_data)
        
    # Align row/column
    temp = np.transpose(np.asarray(temp))        
    # Normalize to [0,1]
    temp = (temp + 1)*0.5
    # Stack the generated data
    data.append(temp)
                
  return data

def ypr_to_quat(yaw, pitch, roll):
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_Angles_to_Quaternion_Conversion
    # Yaw then pitch then roll only!
    yaw = yaw / 180. * math.pi
    pitch = pitch / 180. * math.pi
    roll = roll / 180. * math.pi
    cy = math.cos(yaw * 0.5);
    sy = math.sin(yaw * 0.5);
    cp = math.cos(pitch * 0.5);
    sp = math.sin(pitch * 0.5);
    cr = math.cos(roll * 0.5);
    sr = math.sin(roll * 0.5);

    w = cr * cp * cy + sr * sp * sy;
    x = sr * cp * cy - cr * sp * sy;
    y = cr * sp * cy + sr * cp * sy;
    z = cr * cp * sy - sr * sp * cy;

    return [w, x, y, z]

def qdist(q1,q2):
    return(sum(abs(q1[i] - q2[i]) for i in range(4)))

def data_to_quat(dataset):
  newdata = []
  for i in range(dataset.shape[0]):
      quat = ypr_to_quat(dataset[i,0], dataset[i,1], dataset[i,2])
      otherquat = [val * -1 for val in quat]
      if (i > 0 and qdist(newdata[-1], quat) > qdist(newdata[-1], otherquat)):
          quat = otherquat
      newdata.append(quat)
  return np.array(newdata)

def ypr_to_continuous(dataset):
    for i in range(1, dataset.shape[0]):
        ranges = [360,180,360]
        for j in range(3):
            while dataset[i,j] - dataset[i-1,j] < -ranges[j]/2:
                dataset[i,j] += ranges[j]
            while dataset[i,j] - dataset[i-1,j] > ranges[j]/2:
                dataset[i, j] -= ranges[j]
            if abs(dataset[i,j] - dataset[i-1,j]) > 90:
                return None
    return dataset

def real_data_loading (data_name, seq_len):
  """Load and preprocess real-world datasets.
  
  Args:
    - data_name: stock or energy
    - seq_len: sequence length
    
  Returns:
    - data: preprocessed data.
  """  
  assert data_name in ['stock','energy','head']
  
  if data_name == 'stock':
    ori_datas = [np.loadtxt('data/stock_data.csv', delimiter = ",",skiprows = 1)]
  elif data_name == 'energy':
    ori_datas = [np.loadtxt('data/energy_data.csv', delimiter = ",",skiprows = 1)]
  elif data_name == 'head':
    (_, _, filenames) = next(os.walk('data/set2/'))
    ori_datas = [np.loadtxt('data/set2/'+file, delimiter = ",",skiprows = 1) for file in filenames]
    #ori_datas = [data_to_quat(ori_data) for ori_data in ori_datas]
    # ori_datas = [ypr_to_continuous(ori_data) for ori_data in ori_datas]
    ori_datas = [ori_data[::2,7:10] for ori_data in ori_datas]

  # Flip the data to make chronological data
  # ori_data = ori_data[::-1]
  # Normalize the data
  #ori_datas = [MinMaxScaler(ori_data) for ori_data in ori_datas]
    
  # Preprocess the dataset
  temp_data = []    
  # Cut data by sequence length
  goods = 0
  bads = 0
  for ds,ori_data in enumerate(ori_datas):
    for i in range(0, len(ori_data) - seq_len, 20):
      _x = ori_data[i:i + seq_len].copy()
      #TEMP CODE CHANGE FEATURES
      #_x = _x[:,3:4]
      good = True

      for ft_id in [0,3]:
          continue
          _x[:,ft_id] = ((_x[:,ft_id] + 2)%2)-1
          for idx in range(_x.shape[0]-1):
              if abs(_x[idx,ft_id]) > 0.8 and abs(_x[idx+1,ft_id] - _x[idx,ft_id])>1.5:
                  _x[idx+1:,ft_id] += 2 if _x[idx+1,ft_id] < 0 else -2
                  print("ROTATED", np.min(_x), np.max(_x))
                  #good = False
      _x = ypr_to_continuous(_x)
      
      #if good:
      if _x is not None:
          temp_data.append(_x)
          goods += 1
      else:
          bads += 1
      #print(np.min(_x), np.max(_x))
  #temp_data = np.array(temp_data)
  #print("HERE", np.min(temp_data), np.max(temp_data))
  #temp_data = temp_data+2
  #print("HERE", np.min(temp_data), np.max(temp_data))
  #temp_data = temp_data % 2
  #print("HERE", np.min(temp_data), np.max(temp_data))
  #temp_data = temp_data - 1
  #print("HERE", np.min(temp_data), np.max(temp_data))
  #print("DONE")
  # Mix the datasets (to make it similar to i.i.d)
  print("GOOD BAD", goods, bads)
  idx = np.random.permutation(len(temp_data))    
  data = []
  for i in range(len(temp_data)):
    data.append(temp_data[idx[i]])
  #scaler = StandardScaler()
  #data = np.reshape(np.array(data), (-1,4))
  #data += 1
  #pt = PowerTransformer(method='yeo-johnson', standardize=True)
  #qt = QuantileTransformer(output_distribution='normal', random_state=0)

  #data = data[:,3:4]
  #print("Shap", scs.shapiro(data))
  #data = qt.fit_transform(data)
  #data = qt.inverse_transform(data)
  #data = pt.fit_transform(data)
  #data = scaler.fit_transform(data)
  #print("Shap", scs.shapiro(data))
  #data = np.reshape(data, (-1,temp_data[0].shape[0], 4))
  return data
