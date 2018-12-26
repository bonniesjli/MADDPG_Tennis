## Project Report
### Learning Algorithm
This project uses MADDPG - Multi Agent Deep Deterministic Policy Gradient. <br>

This algorithm uses the framework of centralized training with decentralized execution as described by the original paper. Both agents have its own Actors and Critics. The Actors take into only its agent's own state observation as input, whereas the Critics takes into the full state observation and both agents' actions as input. The approach of multi-agent decentralized actor with centralized critic is illustrated below:

![Alt Text](link)

#### Model Architecture
ACTOR NETWORK: <br>
* Input layer is equal to the states size = 24<br>
* First hidden layer consists of 256 nodes<br>
* Second hidden layer consists of 128 nodes<br>
* Output layer is equal to the action size = 2<br>

CRITIC NETWORK: <br>
* Input layer is equal to the full observation size = (24 + 2)* 2 = 52 <br>
* First hidden layer consists of 256 nodes<br>
* Second hidden layer consists of 128 nodes<br>
* Output layer is equal to 1<br>

#### Hyperparameters
* Discount rate (GAMMA) = 0.99<br>

* Frequecy in updating the network = 1<br>

* Buffer size = 3e5<br>
* Minibatch size = 256<br>
* Actor Learning rate = 1e-4<br>
* Critic Learning rate = 1e-3<br>

* Update rate of target parameter = 1e-3<br>
* Weight decay = 0

### Plot of Rewards

![Alt Text](link)

### Ideas for Future Work
* Exploration is a very important aspect in solving this environment. For my implementation, I added fully random exploration for the first 1200 episodes and partly random exploration for the following 900 episodes. Ideas in working towards better exploration includes:
1. Prioritized Experience Replay. 
2. Some form of meta learning in learning an exploration reward. This could be very interesting given some of the recent work done in meta-learning for exploration and the non-stationary nature of this environment. 
* Hyperparameter optimization. I am interested in understanding more in the effects of each hyperparameters to the stability of training, particularly the effect of TAU. 
* Collaboration and Competition: I am also very interested in generalizing this algorithm to the Soccer environment where both collaboration and competition need to be taken into account. 