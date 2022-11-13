model = keras.models.load_model(model_dir,compile=False)

import gym

seed = 42

# Use the Baseline Atari environment because of Deepmind helper functions
env = make_atari("BreakoutNoFrameskip-v4")
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

video_dir = "/content/drive/MyDrive/game_ai/videos"
env = gym.wrappers.Monitor(env,video_dir,video_callable= lambda episode_id:True, force = True)

n_episodes = 10
returns = []

for i in range(n_episodes):
  ret = 0
  state = np.array(env.reset())
  done = False
  while(not done):
    # Predict action Q-values
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, 0)
    state_tensor = np.array(state_tensor)
    action_probs = model.predict(state_tensor)
    # Take best action
    action = tf.argmax(action_probs[0]).numpy()
    # Update the environment
    next_state , reward, done, _ = env.step(action)
    state = next_state
    
    ret += reward
  returns.append(ret)
env.close()

print("Returns: {}".format(returns))

# **Q4**

import matplotlib.pyplot as plt

episode_reward_history = np.load("/content/drive/MyDrive/game_ai/results/reward_history.npy")
mean_rewards = np.convolve(episode_reward_history, np.ones(100)/100, mode="valid")
plt.plot(mean_rewards)
