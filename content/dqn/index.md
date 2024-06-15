---
title: "Implementing Deep Q-Network: a Reinforcement Learning Beginner's Challenges and Learnings"
date: '2024-06-15'
category: 'blog'
---

Deep Q-Network (DQN) is an Reinforcement Learning (RL) algorithm developed by Mnih et al. at DeepMind in 2013, which combines the classic RL algorithm Q-Learning with deep neural networks. It is considered as the first major success story of deep reinforcement learning due to its super-human level performance in Atari games -- it takes only raw pixels as input (i.e., it "sees" exactly what a human sees), and is able to outperform humans in a wide range of Atari games.

In the last few weeks, I implemented the DQN algorithm from scratch, and managed to match the DeepMind paper's performance in the Atari games of Pong and Breakout. It was the first time I implement an RL paper, and the process was nothing short of surprises -- I ran into all sorts of testing and debugging challenges and it took me a lot longer than I had expected to get the algorithm to work. It was totally worth it though, as I learned so much about RL engineering, and I suspect that a lot of the learnings are applicable to Machine Learning (ML) engineering in general (will verify this when I implement a non-RL ML paper). In this blog post, I'll share the challenges I faced and what I learned as a beginner, together with some tactical tips for debugging DQN on Atari Pong / Breakout.

<style>
.container {
  text-align: center;
  margin-bottom: 20px;
}
.sub-container {
  display: flex;
  justify-content: center;
  gap: 10px;
  flex-wrap: wrap;
}
.caption {
  font-size:80%;
  font-style:italic;
}
.chart {
  height: 250px;
}
</style>

<div class="container">
  <div class="sub-container">
    <div>
      <video controls autoplay loop muted>
        <source src="videos/pong-intro.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <div class="caption">Atari Pong</div>
    </div>
    <div>
      <video controls autoplay loop muted>
        <source src="videos/breakout-intro.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <div class="caption">Atari Breakout</div>
    </div>
  </div>
</div>

The rest of the post assumes familiarity with the DQN algorithm. Take a look at [PyTorch's DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html) for a quick refresher, or refer to the [DeepMind paper](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) for a deep dive.

## Learning 1: building iteratively is the key to testability and debuggability

Testing was my first challenge. After finishing implementation, which itself was pretty straightforward, I suddenly realized that I didn't have a good test plan. The most straightforward approach would be to throw the code to a GPU, wait for 10+ hours, and see if a high performing model is spit out, but that'd be pretty costly both in terms of time (10+ hours per test) and money (at least a couple of dollars to rent a Cloud GPU). I really wanted to first sanity check the code at least to make sure there's no obvious bugs before kicking off the full training process, but struggled to come up with a good idea. The key challenge here is that the DQN algorithm is inherently non-deterministic, especially with the heavy exploration stage in the beginning. So, if you observe the agent's behavior in the first few minutes or even hours and see that it takes unreasonable actions, it's hard to tell if that's due to expected random exploration or there's a bug in the code. To be fair, once you've successfully made a particular algorithm work for a particular problem, you will see patterns of successful / failed runs and can use them as a reference, but that doesn't help you when you work on an algorithm / problem for the first time, which was where I was. Without a better idea, I ended up adding a lot of unit tests and manually examining extremely detailed logs of key components' internal states. It was super tedious, but did give me a lot more confidence in my implementation. Despite that effort though, the training wasn't successful, as shown by the fluctuating yet overall downward trending training reward and an agent that stayed still at the top of the screen.

<div class="container">
  <div class="sub-container">
    <div>
      <img src="images/training-reward-not-learning.png" style="width: auto; height: 210px;">
      <div class="caption">Training reward treanding downwards</div>
    </div>
    <div>
      <video controls autoplay loop muted>
        <source src="videos/1st-failed-run.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <div class="caption">Agent learned to stay still</div>
    </div>
  </div>
</div>

Debugging became the next challenge. I had only two pieces of information that are completely the opposite --  on the one hand, the fairly thorough sanity check passed; on the other hand, after 5+ hours of training, the agent would take actions that were even worse than random -- but nothing in between. With so many components interacting with each other, there were tons of possibilities of what could've gone wrong. For example, some tricky data structures, such as the memory-efficient circular array implementation of the replay buffer, could be buggy even if all unit tests were passed. I might have been misinterpreting the meaning or settings of some hyperparameters from the paper, as it seemed to have used slightly different terminologies from the norm today. The list goes on and on. Again, without a better idea, I tried to log and observe as much information as I possibly could, but this time, it wasn't even remotely helpful. For example, I plotted the state that the agent sees at random steps and how the model weight distributions shift over time, but wasn't able to get any useful insights out of it.

<div class="container">
  <div class="sub-container">
    <div>
      <img src="images/state-action-plot.png" style="width: auto; height: 210px;">
      <div class="caption">The plot of the state (frame 0) at a random step</div>
    </div>
    <div>
      <img src="images/model-weights.png" style="width: auto; height: 210px;">
      <div class="caption">Shfit in the distribution of target net's conv1.bias</div>
    </div>
  </div>
</div>

A quick web search showed me that I wasn't alone. RL seems so notoriously hard to debug that not only beginners struggle with it (e.g., help-seeking posts on [reddit](https://www.reddit.com/r/reinforcementlearning/comments/fw3s1n/how_long_does_training_a_dqn_take/) and [stackoverflow](https://stackoverflow.com/questions/54371840/dqn-stuck-at-suboptimal-policy-in-atari-pong-task)), even experienced researchers have found it tricky as well (e.g., [Alex Irpan's blog post](https://www.alexirpan.com/2018/02/14/rl-hard.html), [Andrej Kaparthy's HackerNews comment](https://news.ycombinator.com/item?id=13519044)). The following two resources were particularly helpful and inspired me to adopt a completely different approach.

1. [The nuts and bolts of deep RL research](https://www.youtube.com/watch?v=8EcdaCk9KaQ) by John Schuman (and its lecture notes named [DeepRLHacks](https://github.com/williamFalcon/DeepRLHacks))
2. [Lessons Learned Reproducing a Deep Reinforcement Learning Paper](http://amid.fish/reproducing-deep-rl) by Amid Fish

Specifically, the following iterative approach seems like a solid strategy for developing/implementing an RL algorithm.
1. Start with a minimal implementation and make it work for an easy problem. An easy problem should give a ton of fast feedback, both in terms how frequently it gives the agent rewards, and how fast it allows the model to train (i.e., solvable by models with few parameters). A minimal implementation should be so simple that it's almost verifiable by just staring at the code. This combination minimizes room for mistakes and allows for fast iterations.
2. Gradually add components that are optional for the easy problem but are essential to the target problem, and use the performance on the easy problem as a test case. In other words, if performance regresses after a component is added, very likely a bug is introduced.
3. Come up with a simplied version of the target problem, and use it for hyperparameter tuning and testing components that are exclusive to the target problem (i.e., those not applicable to the easy problem and hence can't be tested on the easy problem). Once again, the goal of simplification here is to increase iteration speed.
4. Once the simplified version of the target problem is solved, move on to finally solve the original target problem.

This iterative approach makes sense because it breaks down the whole solution into isolated debuggable iterations. When a particular iteration starts to fail, you can attribute the issues to the few limited changes introduced in that iteration, which is much easier to debug than trying to reason across all components that interact in complex ways.

The specific strategy I eventually ended up using looks like this.
1. Use [Cart Pole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) as the easy problem to develop and test a minimal implementation. It has simple states, and gives frequent rewards (+1 for every step when the game is live).
2. Add in more complex components (e.g., circular array implementation of the replay buffer, stacking the last 4 observations as a single state, ability to run a GPU, etc) one by one and make sure the performance on Cart Pole is maintained after each component is added.
3. Use a simplified version of Pong (scores at the top cropped out + count each rally as an episode) to test Atari specific components (e.g., Convolutional Neural Nets) and tune hyperparameters.
4. Hyperparameter tuning for the real Pong and Breakout games.

<div class="container">
  <div class="sub-container">
    <div>
      <video controls autoplay loop muted, style="width: auto; height: 210px;">
        <source src="videos/cart-pole.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <div class="caption">Cart Pole: pushing the cart left/right to keep the pole upright</div>
    </div>
  </div>
</div>

## Learning 2: experimentation won't be efficient without hypotheses
To execute on the first step of the above strategy, my goal was to build a minimal implementation that matches a reasonable performance benchmark. Specifically, I wanted to make my testing reward curve match the one from a [reference blog post](https://medium.com/@ym1942/deep-q-learning-dqn-tutorial-with-cartpole-v0-5505dbd2409e). In this process, I (re)learned the importance of hypothesis-driven experimentation.

<div class="container">
  <div class="sub-container">
    <div>
      <img class="chart" src="images/cart-pole-initial.png">
      <div class="caption">Testing reward curves from my 5 runs</div>
    </div>
    <div>
      <img class="chart" src="images/cart-pole-reward-ref.png">
      <div class="caption">Reference reward curve</div>
    </div>
  </div>
</div>

The reason why I say "relearn" is that I've actually learned and helped many other people learn this principle in a different context of growth/product engineering -- whenever you run an A/B test about a product change, you need to have an hypothesis so that even if the experiment "fails", it gives you insights that help you set better direction going forward. Trial and error rarely gets you anywhere. However, after getting lost in this new RL context where there are so many knobs to tune and there's no clear direction, it's very tempting to just try your luck and see what happens, which got me into this trial and error mode for a few days in the beginning, until I saw the following paragraph from [Amid Fish's blog post](http://amid.fish/reproducing-deep-rl).

> Switching from experimenting a lot and thinking a little to experimenting a little and thinking a lot was a key turnaround in productivity. When debugging with long iteration times, you really need to pour time into the hypothesis-forming step - thinking about what all the possibilities are, how likely they seem on their own, and how likely they seem in light of everything you’ve seen so far. Spend as much time as you need, even if it takes 30 minutes, or an hour. Reserve experiments for once you’ve fleshed out the hypothesis space as thoroughly as possible and know which pieces of evidence would allow you to best distinguish between the different possibilities.

That was a wake up call that made me realize that the same experimentation principle applies to RL as well. After adopting a hypothesis-driven approach, my debugging onf Cart Pole was greatly accelerated. Here's how I did it.

My reward chart shows two issues
1. Testing reward peaked at ~150 for most runs, far below the maximum reward of 200
2. Training was extremely unstable.

I hypothesized that they were due to small batch size (32). I tested increasing batch size to 64 and 96, and the result validated the hypothesis.

<div class="container">
  <div class="sub-container">
    <div>
      <img class="chart" src="images/cart-pole-batch_size=64.png">
      <div class="caption">Batch size = 64</div>
    </div>
    <div>
      <img class="chart" src="images/cart-pole-batch_size=96.png">
      <div class="caption">Batch size = 96</div>
    </div>
  </div>
</div>

Now, with a batch size of 96, near optimal performance was achieved at some point in all the 5 runs. There was still a performance dip betweeen 60k to 70k training steps though. Noticing that (a) the pole is almost always upright when the agent performs well, and (b) 60k steps is when the replay buffer run out of experiences collected in the total random exploration phase, I made the following hypothesis: model training after step 60k was predominantly based on experiences where the pole was upright, which made the model over-optimize for that particular scenario and made the agent to "forget" how to handle other situations. This hypothesis also could explain why its performance picked up again later -- because the replay buffer started to collect sufficiently diverse experiences again after it failed a lot between 60k-70k steps.

<div class="container">
  <div class="sub-container">
    <div>
      <video controls autoplay loop muted, style="width: auto; height: 210px;">
        <source src="videos/cart-pole-upright.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <div class="caption">The pole is almost always upright when the agent performs well</div>
    </div>
  </div>
</div>

I tested increasing the random exploration window from the first 10k steps to the first 20k steps, and that moved the dip from 60k-70k steps to 70k-80k steps, whcih validated my hypothesis.

<div class="container">
  <div class="sub-container">
    <div>
      <img class="chart" src="images/cart-pole-learning_start=20k.png">
      <div class="caption">20k steps of initial total random exploration</div>
    </div>
  </div>
</div>

Since the performance dip was caused by a lack of diverse experiences in the training data, it seemed reasonable to hypothesize either or both of the following ideas can eliminate the dip.
1. Make the model less sensitive to skewed data by increasing the model's capacity.
2. Keep replenising diverse experiences to the replay buffer by maintain a high exploration rate throughout the training process.

With additional experimentation, I concluded that hypothesis 1 was wrong and hypothesis 2 was right -- increased model capcity (4 layers -> 5 layers) helped further stabilize training and postponing the dip a little bit but didn't eliminate it, whereas keeping an exploration rate as high as 90% throughout helps keep testing reward consistently high. 

<div class="container">
  <div class="sub-container">
    <div>
      <img class="chart" src="images/cart-pole-batch_size=96,layer=5.png">
      <div class="caption">Using a 5 layer Multi-Layer Perceptron (MLP) </div>
    </div>
    <div>
      <img class="chart" src="images/cart-pole-eps=0.9.png">
      <div class="caption">End to end 90% exploration rate</div>
    </div>
  </div>
</div>

Now that the algorithm performed reasonably well and that I was able to explain the test results, I became fairly confident about the correctness of my implementation. So, I concluded that additional time spent on Cart Pole would have diminishing returns and decided to swtich back to Atari Pong. As we will see in the next learning, it turns out that I missed a bug here that hinders Adam optimizer's performance (hypothesis 1 above was invalidated only conditioned on such performance limits). With a correct Adam usage and a different way of increasing model capacity (increasing width as opposed to depth), a near-200 testing reward can be consistently maintained even with low exploration rate. My additional learning here is that while hypothesis-driven experimentation can speed up debugging process when you see a bug / performance gap, you can't rule out the possibility of bugs just because the test results are explainable.

<div class="container">
  <div class="sub-container">
    <img class="chart" src="images/cart-pole-adam-correct.png">
  </div>
  <div class="caption">Achieving near 200 testing reward consistently with correct Adam usage and a 3-layer MLP with 128 neurons in the hidden layer </div>
</div>

## Learning 3: it's not really working until results can be consistently reproduced
After switching from solving Cart Pole to solving Pong, a big challenge was getting model training to converge. I tried quite a few combinations of batch size, learning rate, optimization algorithms, gradient clipping, and most of the time I ran into exploding gradients and monotonically changing Q values that don't converge. Here are some examples of what gradient norm, loss and Q value time series looked like for divergent training.

<div class="container">
  <div class="sub-container">
    <img class="chart" src="images/legacy-exp-24-bad-run-meta.png">
    <img class="chart" src="images/legacy-exp-29-bad-run-meta.png">
    <img class="chart" src="images/legacy-exp-30-bad-run-meta.png">
    <img class="chart" src="images/legacy-exp-31-bad-run-meta.png">
    <img class="chart" src="images/legacy-exp-33-bad-run-meta.png">
    <img class="chart" src="images/legacy-exp-34-bad-run-meta.png">
  </div>
  <div class="caption">Some examples of non-convergent training</div>
</div>

Eventually, I found a sweet spot of batch size 96 and learning rate annealing from 7.5e-5 to 7.5e-6 in 2.5 million steps. This setting not only helped model training converge, but also, when combined with some other hyperparameter tweaks, helped match the paper's result on Pong performance. I ran it multiple times to validate that it was reproducible, and they all worked. As I was happily cleaning up the code and preparing to publish it on Github, something strange happened -- model training suddenly failed to converge again! Initially, I thought I must have introduced some bugs as part of the refactoring, but that was quickly ruled out because even the same code revision that used to work was no longer working. Then I realized that I never explicitly set any random seed, so by default, some combination of system time and other system environment factors were used as random seed. That explained why the training behaved completely differently when it was run on Monday v.s. Friday.

After putting in explicit random seeds, I ran the training again and something even stranger happened -- with the same code, same hyperparameter setting and same random seed, training results were completely different when it was run on an Nvidia RTX 3090 v.s. a RTX 4090! The only explaination I could think of was that maybe the two kinds of GPUs handle floating point numerical computations slightly differently, which is negligible normally but in this case got amplified in the model traning process.

<div class="container">
  <div class="sub-container">
    <img class="chart" src="images/pong-rtx3090.png">
    <img class="chart" src="images/pong-rtx4090.png">
  </div>
  <div class="caption">Comparison between two runs on RTX 3090 (top) v.s. two runs on RTX 4090 (bottom). Between two runs on the same GPU, the curves almost overlap perfectly (it looks like there's just one curve in each box but there are two), whereas training converges on RTX 3090 and diverges on RTX 4090</div>
</div>

After anther round of experimentation, I found that switching learning rate schedule from linear anneal to exponential decay solved the problem. For a while, I was amazed by how much a small change in learning rate schedule can make or break an algorithm, but later I realized that was not the only thing changed -- I also switched my self-implemented learning rate scheduler to one of PyTorch's official schedulers, and in that process I changed how PyTorch optimizers are used. Previously, a new optimizer would be created in every training step, whereas in order to use PyTorch's official learning rate scheduler, I was forced to use the same optimizer throughout the whole training process. It turns out the previous usage was problematic, because Adam optimizer's moment history gets lost whenever a new optimizer instance is created, and that significantly impacts performance. So eventually, I found a long standing bug that was only surfaced when I looked into reproducibility issues.

<div class="container">
  <div class="sub-container">
    <img class="chart" src="images/exponential-decay-lr-8runs.png" style="height: 500px;">
  </div>
  <div class="caption">8 runs with correct usage of Adam optimizer (4 runs on RTX 3090 / 4090 each)</div>
</div>

## Learning 4: small reward differences can lead to big performance gaps
Once I was able to make model training converge, the next step was to continuously improve the agent's performance until it matches the paper's results. Here, I learned how small differences in the reward mechanism could lead to huge performance gaps.

I hit the first performance bottleneck when I was working on the simplified version of Pong (cropping out the scores at the top and count each rally as an episode using a customly built Gym environment wrapper), where I had a hard time getting the score to go above 0. Two unexpected observations caught my eye -- (1) there were quite a few unexpected episode length spikes (see the chart below), and (2) I saw some saved models with a testing reward of 18, which shouldn't have been possible since simpfied Pong game has a testing reward in the range of [-1, 1]. (Fortunately I saved every model that broke the testing reward record and saved the record score as part of model name.) The video recording of the reward-18 model explained the situation -- the agent learned to drag the game for such a long time that it would eventually break the environment (see videos below). Additionally, I found that right when the environment got broken, a reward of 4 would be returned. Here's my hypothesis of what was going on. Since it was hard to beat the opponent, the agent first learned to delay the loss of a score as much as possible as opposed to learning to win a score. Then, as it delayed the loss longer and longer, it learned that if it delayed the loss long enough, it would even get a lot more rewards (4) than winning a score (1)! This dynamics kept the agent stuck in this local optimum.

I was able to get over this bottleneck when I artificially truncate an episode long before the environment is broken.

<div class="container">
  <div class="sub-container">
    <div>
      <img src="images/easy-pong-v2-exp5-super-long-episodes.png" style="width: auto; height: 210px;">
      <div class="caption">Testing reward was at 0, and there were unexpected eposide length spikes </div>
    </div>
    <div>
      <video controls autoplay loop muted>
        <source src="videos/easy-pong-v2-exp-6-stuck.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <video controls autoplay loop muted>
        <source src="videos/easy-pong-v2-exp-6-broken.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <div class="caption">Agent learned to drag the game for as long as possible (left). Eventually it broke the environment (right) and then received a reward of 4</div>
    </div>
  </div>
</div>

The second major performance bottoleneck was hit in Breakout. Testing reward was stuck at around 100 and the agent seemed to be trying its best to clear the bottom three layers of bricks but avoided balls that bounced back from the top three layers. 

<div class="container">
  <div class="sub-container">
    <div>
      <img src="images/breakout-bottleneck.png" style="width: auto; height: 210px;">
      <div class="caption">Testing reward was stuck at around 100</div>
    </div>
    <div>
      <video controls autoplay loop muted>
        <source src="videos/breakout-bottleneck.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <div class="caption">The agent tries to clear up the bottle three layers</div>
    </div>
  </div>
</div>

I was really confused for a while until I saw a hint from the discussion [here](https://github.com/dennybritz/reinforcement-learning/issues/30). The issue was that I treated the whole episode consistenting of 5 lives as a single episode as opposed to treating each life as an episode. Unlike Pong, where there's an immediate negative reward when the opponent scores, Breakout doesn't give the agent any immediate negative reward when it loses a life. Since the agent optimizes for a discounted future reward, it's relatively short sighted -- for a gamma value of 0.99, rewards received after 1000 steps would be upper bounded by 0.99^1000 / (1-0.99) < 0.005 (sum of geometric series), which is negligible. This dynamics doesn't provide enough incentives for the agent to learn to play the harder moves (the ball moves much faster when it's bounced back from the top three layers) to save a life. Treating each life as an episode pushes the agent to learn to get as many rewards in a single life as possible. Later, with additional optimizations and much longer training (50 million steps), I was able to get to a testing score of 340 for a configeraion that doesn't treat each life as an episode. However, it still lags the configuration that treats each life as an episode by ~100 testing reward consistenly in 50 million steps of training. (Maybe eventually they'll converge but I haven't tested it at the moment).

<div class="container">
  <div class="sub-container">
    <img class="chart" src="images/breakout-loss-of-lfe-as-episode-end-or-not.png" style="height: 500px;">
  </div>
  <div class="caption">Performance gap between treating each life as an episode (exp-44) v.s. not (exp-47)</div>
</div>

## Results and additional tips for Pong and Breakout

Overall, after tackling the above challenges, I was able to get the following results with vanilla DQN (no additional techniques such as Double DQN or Dueling DQN were used), which roughly matches those in the DeepMind paper. Testing reward (labeled as eval_reward_0.05.avg in the reference charts below) is calculated by taking the mean of a rolling window of the most recent 50 eval episodes, where an eval eposide is run once every 10k steps, with a 5% exploration rate (*note: testing/eval reward charts in the learnings sections above are based on a 1% exploration*).

- Pong: maximum testing reward of 19 in 6 million steps (v.s. benchmark 18.9 +/- 1.3)
- Breakout: maximum testing reward of 377.6 in 60 million steps (v.s. benchmark 401.2 +/- 26.9)

Here are some video recordings of the best performing models in action.

<div class="container">
  <div class="sub-container">
    <div>
      <video controls autoplay loop muted>
        <source src="videos/pong-21.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <div class="caption">Pong</div>
    </div>
    <div>
      <video controls autoplay loop muted>
        <source src="videos/breakout-486.mp4" type="video/mp4">
        Your browser does not support the video tag.
      </video>
      <div class="caption">Breakout</div>
    </div>
  </div>
</div>

The above results are achieved with the following configuration.

1. Exploration rate: anneals linearly from 100% to 10% in 1 million steps
2. Learning rate: anneals from 7.5e-5 to 7.5e-6 in 5 million steps
3. Batch size: 96
4. Reward discount factor (gamma): 0.99
5. Replay buffer size: 1 million
6. Initial random exploration steps: 100k
7. Target network update freqency: once in every 10k steps
8. Training frequency: once in every 4 steps

All code can be found in the following Github repos.
1. [dqn-cart-pole](https://github.com/mingfei-li/dqn-cart-pole)
2. [dqn-atari](https://github.com/mingfei-li/dqn-atari)

### Additional tips

0. Do not get addicted to watching TensorBoard! This is technically a general tip as opposed to a Pong/Breakout tip but it just can't be stressed enough! As also called out by most blog posts / videos referenced above, it's a big productivity killer and potentially hurts your mental health.
1. If you are also a beginner, I highly recommend following the iterative appraoch mentioned in Learning 1. It might seem slower, but it helps you make steady progress rather than getting completely stuck/lost.
2. I never found gradient clipping useful. In cases where gradient was clipped, it didn't make the training converge but instead pushed the Q value to diverge to the opposite direction. Adam with learning rate annealing should do the job without gradient clipping.
3. If your implementation is correct and hyperparameter setting is reasonable, learning should be fairly fast. For Pong and Breakout, you should see either testing episode length or reward going up steadily in less than 500k steps.
4. It takes a lot of samples and a long time to train for Breakout. For example, it took me 30 million steps to get the average testing reward (5% exploration) to go above 300.
5. Feel free to use the following charts as a reference on what a reasonable training run looks like for Pong and Breakout.
6. Shoot me an email if you need additional help after reading this blog post and all the references.

<div class="container">
  <div class="sub-container">
    <img class="chart" src="images/pong-ref.png" style="height: 800px;">
  </div>
  <div class="caption">Pong Reference</div>
</div>

<div class="container">
  <div class="sub-container">
    <img class="chart" src="images/breakout-ref.png" style="height: 800px;">
  </div>
  <div class="caption">Breakout Reference</div>
</div>