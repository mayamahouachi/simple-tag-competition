# train_ppo_predators.py
import random
from collections import deque
import numpy as np
import torch
from pettingzoo.mpe import simple_tag_v3
from reference_agents_source.prey_agent import StudentAgent as ReferencePreyAgent
from ppo import PPOAgent

PREDATORS = ["adversary_0", "adversary_1", "adversary_2"]
PREY_ID = "agent_0"


def set_all_seeds(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(num_episodes: int = 3000,max_cycles: int = 200,batch_size: int = 10,save_path: str = "predator_model.pth",
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
    epsilon_clip: float = 0.2,
    K_epochs: int = 4,
    gae_lambda: float = 0.95,
    entropy_coef: float = 0.01,
    value_clip: float = 0.2,
    max_grad_norm: float = 0.5):
    env = simple_tag_v3.parallel_env(num_good=1,num_adversaries=3,num_obstacles=2,max_cycles=max_cycles,continuous_actions=False)
    obs, _ = env.reset(seed=0)
    state_dim = env.observation_space(PREDATORS[0]).shape[0]  # 16
    action_dim = env.action_space(PREDATORS[0]).n             # 5

    ppo = PPOAgent(
        state_dim=state_dim,action_dim=action_dim,batch_size=batch_size,lr_actor=lr_actor,lr_critic=lr_critic,gamma=gamma,epsilon_clip=epsilon_clip,K_epochs=K_epochs,
        gae_lambda=gae_lambda,entropy_coef=entropy_coef,value_clip=value_clip,max_grad_norm=max_grad_norm, n_parallel=3)
    ppo.set_training(True)

    prey = ReferencePreyAgent()
    scores = deque(maxlen=100)

    print(
        f"Training PPO predators vs reference prey | "
        f"state_dim={state_dim}, action_dim={action_dim} | "
        f"lr_actor={lr_actor:g}, lr_critic={lr_critic:g}, gamma={gamma:g}, "
        f"clip={epsilon_clip:g}, K={K_epochs}, gae={gae_lambda:g}, ent={entropy_coef:g}, "
        f"vclip={value_clip:g}, gnorm={max_grad_norm:g}, batch={batch_size}"
    )

    for ep in range(num_episodes):
        set_all_seeds(ep)
        obs, _ = env.reset(seed=ep)
        ep_pred_sum = 0.0

        for t in range(max_cycles):
            actions = {}

            for pid in PREDATORS:
                actions[pid] = ppo.select_action(obs[pid], greedy=False)

            actions[PREY_ID] = int(prey.get_action(obs[PREY_ID], PREY_ID))

            obs, rewards, terminations, truncations, infos = env.step(actions)

            for pid in PREDATORS:
                r = float(rewards[pid])
                ppo.store_reward(r)
                ep_pred_sum += r

            done_pred = all(terminations.get(pid, False) or truncations.get(pid, False)for pid in PREDATORS)
            if done_pred:
                break

        ppo.end_episode(next_state=None, done=True)

        scores.append(ep_pred_sum)
        if (ep + 1) % 50 == 0:
            print(f"Ep {ep+1:4d} | avg100 predator sum reward: {np.mean(scores):.3f}")

    ppo.save(save_path)
    print(f"Saved in {save_path}")
    env.close()


if __name__ == "__main__":
    # train(
    #     num_episodes=3000,
    #     max_cycles=200,
    #     batch_size=10,
    #     save_path="submissions/mayamahouachi/predator_model.pth",
    # )
    train(
        num_episodes=5000,
        max_cycles=200,
        batch_size=30,
        save_path="submissions/mayamahouachi/new_best_shared_predator_model.pth",
        lr_actor=0.0004739216012260275,
        lr_critic=0.004715828399412282,
        gamma=0.9517602627703191,
        epsilon_clip=0.2917398527388533,
        K_epochs=8,
        gae_lambda=0.9897753741537167,
        entropy_coef=0.010693130119891094,
        value_clip=0.16511571907917394,
        max_grad_norm=0.7456998598553036,
    )


# Best params: {'lr_actor': 0.0004739216012260275, 'lr_critic': 0.004715828399412282, 'gamma': 0.9517602627703191, 'epsilon_clip': 0.2917398527388533, 'K_epochs': 8, 'gae_lambda': 0.9897753741537167, 'entropy_coef': 0.010693130119891094, 'batch_size': 30, 'value_clip': 0.16511571907917394, 'max_grad_norm': 0.7456998598553036}