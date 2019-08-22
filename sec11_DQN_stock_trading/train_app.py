from agent import Agent 
from market_env import Market
import os
import time 
from collections import Counter

def main():
    window_size = 5
    eposide_cnt = 2
    stock_name = "GSPC_2011"
    batch_size = 32
    profit_for_episode = []
    total_action_history = []
    agent = Agent(window_size)
    market = Market(window_size, stock_name)

    start_time = time.time()
    for e in range(1, eposide_cnt + 1):
        print("Episode {}/{}".format(e, eposide_cnt))
        agent.reset()
        state, price_data = market.reset()
        
        for t in range(market.last_data_index):
            action, bought_price = agent.act(state, price_data)
        
            next_state, next_price_data, reward, done = \
                        market.get_next_state_reward(action, bought_price)
            agent.memory.append((state, action, reward, next_state, done))
    
            if len(agent.memory) > batch_size:
                agent.exprience_replay(batch_size)
    
            state = next_state
            price_data = next_price_data
    
            if done:
                print("--------------------------------")
                print("Total profit: {}".format(agent.get_total_profit()))
                print("action history")
                print(Counter(agent.action_history).keys())
                print(Counter(agent.action_history).values())
                total_action_history.append(agent.action_history)
                print("--------------------------------")
                profit_for_episode.append(agent.get_total_profit())

        if e % 10 == 0:
            if not os.path.exists("models"):
                os.mkdir("models")
            print(str(e))
            agent.model.save("models/model_ep{}.h5".format(str(e)))

    end_time = time.time()
    training_time = end_time - start_time
    print("Training time took {:.2f} seconds.".format(training_time))
    print("profit_for_episode = ", profit_for_episode)
    print("total action history ")
    for history in total_action_history:
        print(Counter(history).keys())
        print(Counter(history).values())

if __name__ == "__main__":
    main()
