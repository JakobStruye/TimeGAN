"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
"Time-series Generative Adversarial Networks," 
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: April 24th 2020
Code author: Jinsung Yoon (jsyoon0823@gmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

# Necessary Packages
import tensorflow as tf
import numpy as np
from utils import extract_time, rnn_cell, random_generator, batch_generator
import pickle
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import random

def generate_data(sess, no, z_dim, ori_time, max_seq_len, X_hat, Z, X, T, Z_mb, ori_data, mms, qt):
  ## Synthetic data generation
  # multiplier = 1
  # no *= multiplier
  # ori_time = multiplier * [1 * val for val in ori_time]
  # max_seq_len *= multiplier
  Z_mb = random_generator(no, z_dim, ori_time, max_seq_len)
  generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})

  generated_data = list()
  for i in range(no):
    temp = generated_data_curr[i, :ori_time[i], :]
    generated_data.append(temp)

  # Renormalization
  gen_np = np.array(generated_data)
  generated_data = np.array(generated_data)
  print("MINMAX", np.min(gen_np), np.max(gen_np))
  #generated_data = [entry + (0.5 + 1e-10) for entry in generated_data]
  #generated_data = [entry % 1.0 for entry in generated_data]
  #generated_data = generated_data * max_val
  #generated_data = generated_data + min_val
  gen_2d = np.reshape(generated_data, (-1,generated_data.shape[2]))
  gen_2d = mms.inverse_transform(gen_2d)
  gen_2d = qt.inverse_transform(gen_2d)
  generated_data = np.reshape(gen_2d, generated_data.shape)
  print("MINMAX", np.min(np.array(generated_data)), np.max(np.array(generated_data)))
  return generated_data

def timegan (ori_data, parameters):
  """TimeGAN function.
  
  Use original data as training set to generater synthetic data (time-series)
  
  Args:
    - ori_data: original time-series data
    - parameters: TimeGAN network parameters
    
  Returns:
    - generated_data: generated time-series data
  """
  randval = str(random.randint(100000,999999))
  print("RANDVAL", randval)
  # Initialization on the Graph
  tf.reset_default_graph()

  # Basic Parameters
  no, seq_len, dim = np.asarray(ori_data).shape
    
  # Maximum sequence length and each sequence length
  ori_time, max_seq_len = extract_time(ori_data)
  
  '''def MinMaxScalerNO(data):
    """Min-Max Normalizer.
    
    Args:
      - data: raw data
      
    Returns:
      - norm_data: normalized data
      - min_val: minimum values (for renormalization)
      - max_val: maximum values (for renormalization)
    """ 
    floor = 0.25
    ceil = 0.75
    max_val = np.max(np.max(data, axis = 0), axis = 0) / (ceil + floor)
    data = data / (max_val + 1e-7)
    
    min_val = np.min(np.min(data, axis = 0), axis = 0) - floor
    
    data = data - min_val
      
      
    return data, min_val, max_val
  '''

  # Normalization
  ori_data = np.array(ori_data)
  print("before", np.min(ori_data), np.max(ori_data))
  qt = QuantileTransformer(output_distribution="normal", random_state=0)
  mms = MinMaxScaler(feature_range=(0.0,1.0))
  ori_2d = np.reshape(ori_data, (-1,ori_data.shape[2]))
  ori_2d = qt.fit_transform(ori_2d)
  ori_2d = mms.fit_transform(ori_2d)
  ori_data = np.reshape(ori_2d, ori_data.shape)
  #ori_data, min_val, max_val = MinMaxScaler(ori_data)
  print("after", np.min(ori_data), np.max(ori_data))
  #yes = 0
  #no = 0
  #print(type(ori_data))
  #new_ori_data = []
  #for d in ori_data:
  #    if np.max(d) > 0.82: 
  #        yes+=1
  #        new_ori_data.append(d)
  #    else: 
  #        no +=1
  #ori_data = np.array(new_ori_data)
  #print("YES", yes, "NO", no)
  #ori_data += 0.5 - 1e-10
  #ori_data %= 1.0
  ## Build a RNN networks
  
  # Network Parameters
  hidden_dim   = parameters['hidden_dim'] 
  num_layers   = parameters['num_layer']
  iterations   = parameters['iterations']
  generate_interval = parameters['generate_interval']
  batch_size   = parameters['batch_size']
  module_name  = parameters['module']
  learning_rate = parameters['learning_rate']
  z_dim        = dim
  gamma        = 1

  print("ZDIM", z_dim)
  # Input place holders
  X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
  Z = tf.placeholder(tf.float32, [None, max_seq_len, z_dim], name = "myinput_z")
  T = tf.placeholder(tf.int32, [None], name = "myinput_t")
  
  def embedder (X, T):
    """Embedding network between original feature space to latent space.
    
    Args:
      - X: input time-series features
      - T: input time information
      
    Returns:
      - H: embeddings
    """
    with tf.variable_scope("embedder", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, X, dtype=tf.float32, sequence_length = T)
      H = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
    return H
      
  def recovery (H, T):   
    """Recovery network from latent space to original space.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - X_tilde: recovered data
    """     
    with tf.variable_scope("recovery", reuse = tf.AUTO_REUSE):       
      r_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      r_outputs, r_last_states = tf.nn.dynamic_rnn(r_cell, H, dtype=tf.float32, sequence_length = T)
      X_tilde = tf.contrib.layers.fully_connected(r_outputs, dim, activation_fn=tf.nn.sigmoid) 
    return X_tilde
    
  def generator (Z, T):  
    """Generator function: Generate time-series data in latent space.
    
    Args:
      - Z: random variables
      - T: input time information
      
    Returns:
      - E: generated embedding
    """        
    with tf.variable_scope("generator", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, Z, dtype=tf.float32, sequence_length = T)
      E = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
    return E
      
  def supervisor (H, T): 
    """Generate next sequence using the previous sequence.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - S: generated sequence based on the latent representations generated by the generator
    """          
    with tf.variable_scope("supervisor", reuse = tf.AUTO_REUSE):
      e_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(max(1,num_layers-1))])
      e_outputs, e_last_states = tf.nn.dynamic_rnn(e_cell, H, dtype=tf.float32, sequence_length = T)
      S = tf.contrib.layers.fully_connected(e_outputs, hidden_dim, activation_fn=tf.nn.sigmoid)     
    return S
          
  def discriminator (H, T):
    """Discriminate the original and synthetic time-series data.
    
    Args:
      - H: latent representation
      - T: input time information
      
    Returns:
      - Y_hat: classification results between original and synthetic time-series
    """        
    with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE):
      d_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell(module_name, hidden_dim) for _ in range(num_layers)])
      d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, H, dtype=tf.float32, sequence_length = T)
      Y_hat = tf.contrib.layers.fully_connected(d_outputs, 1, activation_fn=None) 
    return Y_hat   
    
  # Embedder & Recovery
  H = embedder(X, T)
  X_tilde = recovery(H, T)
    
  # Generator
  E_hat = generator(Z, T)
  H_hat = supervisor(E_hat, T)
  H_hat_supervise = supervisor(H, T)
    
  # Synthetic data
  X_hat = recovery(H_hat, T)
  X_hat = tf.identity(X_hat, name="X_hat")
    
  # Discriminator
  Y_fake = discriminator(H_hat, T)
  Y_real = discriminator(H, T)     
  Y_fake_e = discriminator(E_hat, T)
    
  # Variables        
  e_vars = [v for v in tf.trainable_variables() if v.name.startswith('embedder')]
  r_vars = [v for v in tf.trainable_variables() if v.name.startswith('recovery')]
  g_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
  s_vars = [v for v in tf.trainable_variables() if v.name.startswith('supervisor')]
  d_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
    
  # Discriminator loss
  D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
  D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
  D_loss_fake_e = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake_e), Y_fake_e)
  D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e
            
  # Generator loss
  # 1. Adversarial loss
  G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
  G_loss_U_e = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake_e), Y_fake_e)
    
  # 2. Supervised loss
  G_loss_S = tf.losses.mean_squared_error(H[:,1:,:], H_hat_supervise[:,:-1,:])
    
  # 3. Two Momments
  G_loss_V1 = tf.reduce_mean(tf.abs(tf.sqrt(tf.nn.moments(X_hat,[0])[1] + 1e-6) - tf.sqrt(tf.nn.moments(X,[0])[1] + 1e-6)))
  G_loss_V2 = tf.reduce_mean(tf.abs((tf.nn.moments(X_hat,[0])[0]) - (tf.nn.moments(X,[0])[0])))
    
  G_loss_V = G_loss_V1 + G_loss_V2
    
  # 4. Summation
  G_loss = G_loss_U + gamma * G_loss_U_e + 100 * tf.sqrt(G_loss_S) + 100*G_loss_V 
            
  # Embedder network loss
  E_loss_T0 = tf.losses.mean_squared_error(X, X_tilde)
  E_loss0 = 10*tf.sqrt(E_loss_T0)
  E_loss = E_loss0  + 0.1*G_loss_S
    
  # optimizer
  print("LR", learning_rate)
  E0_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(E_loss0, var_list = e_vars + r_vars)
  E_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(E_loss, var_list = e_vars + r_vars)
  D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list = d_vars)
  G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list = g_vars + s_vars)
  GS_solver = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss_S, var_list = g_vars + s_vars)
        
  ## TimeGAN training   
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  


  # 1. Embedding network training
  print('Start Embedding Network Training')

  for itt in range(iterations):
    np.random.shuffle(ori_data)
    for b_idx in range(no//batch_size):
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size, b_idx)
      # Train embedder
      _, step_e_loss = sess.run([E0_solver, E_loss_T0], feed_dict={X: X_mb, T: T_mb})
    # Checkpoint
    if itt % 1 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) + ', e_loss: ' + str(np.round(np.sqrt(step_e_loss),4)) )

  print('Finish Embedding Network Training')

  # 2. Training only with supervised loss
  print('Start Training with Supervised Loss Only')

  for itt in range(iterations):
    np.random.shuffle(ori_data)
    for b_idx in range(no//batch_size):
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size, b_idx)
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      # Train generator
      _, step_g_loss_s = sess.run([GS_solver, G_loss_S], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
    # Checkpoint
    if itt % 1 == 0:
      print('step: '+ str(itt)  + '/' + str(iterations) +', s_loss: ' + str(np.round(np.sqrt(step_g_loss_s),4)) )

  print('Finish Training with Supervised Loss Only')

  # 3. Joint Training
  print('Start Joint Training')

  for itt in range(iterations):
    for b_idx in range(no//batch_size):
      # Generator training (twice more than discriminator training)
      for kk in range(2):
        np.random.shuffle(ori_data)
        # Set mini-batch
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size, b_idx)
        # Random vector generation
        Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
        # Train generator
        _, step_g_loss_u, step_g_loss_s, step_g_loss_v = sess.run([G_solver, G_loss_U, G_loss_S, G_loss_V], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
         # Train embedder
        _, step_e_loss_t0 = sess.run([E_solver, E_loss_T0], feed_dict={Z: Z_mb, X: X_mb, T: T_mb})
      np.random.shuffle(ori_data)
      # Discriminator training
      # Set mini-batch
      X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size, b_idx)
      # Random vector generation
      Z_mb = random_generator(batch_size, z_dim, T_mb, max_seq_len)
      # Check discriminator loss before updating
      check_d_loss = sess.run(D_loss, feed_dict={X: X_mb, T: T_mb, Z: Z_mb})
      # Train discriminator (only when the discriminator does not work well)
      if (check_d_loss > 0.15):
        _, step_d_loss = sess.run([D_solver, D_loss], feed_dict={X: X_mb, T: T_mb, Z: Z_mb})

    if generate_interval > 0 and (itt+1) % generate_interval == 0:
      generated_data = generate_data(sess, no, z_dim, ori_time, max_seq_len, X_hat, Z, X, T, Z_mb, ori_data, mms, qt)
      pickle.dump(generated_data, open(
        "gendata{}_epoch{}_batch{}_hiddendim{}_numlayer{}_lr{}_intermittent_{}.out".format(randval, iterations, batch_size, hidden_dim, num_layers, learning_rate, itt+1),
        "wb"))
      generated_data = np.vstack([generate_data(sess, no, z_dim, ori_time, max_seq_len, X_hat, Z, X, T, Z_mb, ori_data, mms, qt) for _ in range(10)])
      pickle.dump(generated_data, open(
        "gendata{}_epoch{}_batch{}_hiddendim{}_numlayer{}_lr{}_intermittent_{}BIG.out".format(randval, iterations, batch_size,
                                                                                         hidden_dim, num_layers,
                                                                                         learning_rate, itt + 1),
        "wb"))
      saver = tf.train.Saver()
      saver.save(sess, 'model-{}'.format(itt+1))
      pickle.dump([no, z_dim, ori_time, max_seq_len, ori_data, mms, qt, Z_mb], open("vars{}.pkl".format(itt+1), "wb"))

    # Print multiple checkpoints
    if itt % 1 == 0:
      print('step: '+ str(itt) + '/' + str(iterations) +
          ', d_loss: ' + str(np.round(step_d_loss,4)) +
          ', g_loss_u: ' + str(np.round(step_g_loss_u,4)) +
          ', g_loss_s: ' + str(np.round(np.sqrt(step_g_loss_s),4)) +
          ', g_loss_v: ' + str(np.round(step_g_loss_v,4)) +
          ', e_loss_t0: ' + str(np.round(np.sqrt(step_e_loss_t0),4))  )
  print('Finish Joint Training')

  generated_data = generate_data(sess, no, z_dim, ori_time, max_seq_len, X_hat, Z, X, T, Z_mb, ori_data, mms, qt)
  pickle.dump(generated_data, open("gendata_epoch{}_batch{}_hiddendim{}_numlayer{}.out".format(iterations, batch_size, hidden_dim, num_layers)  , "wb"))
  return generated_data
