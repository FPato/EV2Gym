from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import EvalCallback
from sb3_contrib import TQC, TRPO, ARS, RecurrentPPO

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.rl_agent.reward import SquaredTrackingErrorReward, ProfitMax_TrPenalty_UserIncentives, \
                            V2G_profitmaxV2, ProfitMax_averaged, ProfitMax_TrPenalty_UserIncentives_2, \
                            ProfitMax_SatisfactionFirst
from ev2gym.rl_agent.reward import profit_maximization

from ev2gym.rl_agent.state import V2G_profit_max, PublicPST, V2G_profit_max_loads

import gymnasium as gym

import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import os
import yaml
import matplotlib.pyplot as plt


if __name__ == "__main__":
    config_file = "ev2gym/example_config_files/test-scenario.yaml"
    reward_function = ProfitMax_TrPenalty_UserIncentives_2 #V2G_profitmaxV2
    state_function = V2G_profit_max_loads

    gym.envs.register(id='evs-v0', entry_point='ev2gym.models.ev2gym_env:EV2Gym',
                      kwargs={'config_file': config_file,
                              'verbose': False,
                              'save_plots': False,
                              'generate_rnd_game': True,
                              'reward_function': reward_function,
                              'state_function': state_function,
                              })

    env = gym.make('evs-v0')
    device = "cuda:0"
    algorithm = "sac"

    if algorithm == "ppo":
        model = PPO("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "td3":
        model = TD3("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "sac":
        model = SAC("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")
    elif algorithm == "a2c":
        model = A2C("MlpPolicy", env, verbose=1,
                    device=device, tensorboard_log="./logs/")

    #print(f"SAC actor net_arch: {model.policy.actor.net_arch}")
    #print(f"SAC policy net_arch: {model.policy.net_arch}")
    #print(f"SAC critic qf0: {model.policy.critic.qf0}")
    #print(f"SAC critic qf1: {model.policy.critic.qf1}")

    #[865413, 619614, 712708, 91735, 154548]
    save_path = "saved_models/1cs_V2GProfitPlusLoads/SETSEED_712708_sac_ProfitMax_TrPenalty_UserIncentives_2_V2G_profit_max_loads_v57_open_ae"

    #print(f"PATH: {save_path}/best_model.zip")
    model = model.load(f"{save_path}/best_model", env=env)

    env = model.get_env()
    obs = env.reset()


    stats = []
    all_how_much_charge = []
    all_pv_output = []
    all_ev_soc = []
    current_time = []
    all_actions = []
    all_prices = []

    #for i in range(96 * 100):
    for i in range(96):

        action, _states = model.predict(obs, deterministic=True)
        #print("action: ", action)
        obs, reward, done, info = env.step(action)

        # env.render()
        # VecEnv resets automatically
        if done:
            all_how_much_charge.append(info[0]['info_how_much_charge'])
            all_pv_output.append(info[0]['info_pv_output'])
            all_ev_soc.append(info[0]['info_ev_soc'])
            current_time = info[0]['info_current_time'][:96]
            all_actions.append(info[0]['info_actions'])
            all_prices.append(info[0]['info_prices'])

            stats.append(info)
            obs = env.reset()

    plt.plot(current_time, [sum([x[i] for x in all_how_much_charge])/len(all_how_much_charge)
                            for i in range(len(all_how_much_charge[0]))], label="Charging power (kW)")
    plt.plot(current_time, [sum([x[i] for x in all_pv_output])/len(all_pv_output)
                            for i in range(len(all_pv_output[0]))], label="PV output (kWh)")
    plt.plot(current_time, [sum([x[i] for x in all_ev_soc])/len(all_ev_soc)
                            for i in range(len(all_ev_soc[0]))], label="EV SoC (÷10)")
    plt.plot(current_time, [sum([x[i] for x in all_actions]) / len(all_actions)
                            for i in range(len(all_actions[0]))], label="Action taken")
    #plt.plot(current_time, [sum([x[i] for x in all_prices]) / len(all_prices)
    #                        for i in range(len(all_prices[0]))], label="Electricity prices")

    plt.xticks(current_time[::4], rotation=45, fontsize=8)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Charging power, EV SoC, action taken and PV output over time")

    plt.legend()
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("my_plots/plot_charge_v57_setseed_712708_OPTIMAL.png")

    # print average stats
    print("=====================================================")
    print(f' Average stats for {algorithm} algorithm, {len(stats)} episodes')
    print("total_ev_served: ", sum(
        [i[0]['total_ev_served'] for i in stats]) / len(stats))
    print("total_profits: ", sum(
        [i[0]['total_profits'] for i in stats]) / len(stats))
    print("total_energy_charged: ", sum(
        [i[0]['total_energy_charged'] for i in stats]) / len(stats))
    print("total_energy_discharged: ", sum(
        [i[0]['total_energy_discharged'] for i in stats]) / len(stats))
    print("average_user_satisfaction: ", sum(
        [i[0]['average_user_satisfaction'] for i in stats]) / len(stats))
    print("power_tracker_violation: ", sum(
        [i[0]['power_tracker_violation'] for i in stats]) / len(stats))
    print("tracking_error: ", sum(
        [i[0]['tracking_error'] for i in stats]) / len(stats))
    print("energy_user_satisfaction: ", sum(
        [i[0]['energy_user_satisfaction'] for i in stats]) / len(stats))
    print("total_transformer_overload: ", sum(
        [i[0]['total_transformer_overload'] for i in stats]) / len(stats))
    print("reward: ", sum([i[0]['episode']['r'] for i in stats]) / len(stats))

