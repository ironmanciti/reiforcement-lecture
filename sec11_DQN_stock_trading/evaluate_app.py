from tensorflow.keras.models import load_model
from agent import Agent 
from market_env import Market 
import matplotlib.pyplot as plt

def main():
    stock_name = "GSPC_2011-03"
    model_name = "model_ep10.h5"

    model = load_model("models/" + model_name)
    window_size = model.layers[0].input.shape.as_list()[1]  # 2nd item
    
    agent = Agent(window_size, model_name=model_name)
    market = Market(window_size, stock_name)

    state, price_data = market.reset()

    for t in range(market.last_data_index):
            action, bought_price = agent.act(state, price_data)
        
            next_state, next_price_data, reward, done = \
                        market.get_next_state_reward(action, bought_price)

            state = next_state
            price_data = next_price_data
    
            if done:
                print("--------------------------------")
                print("{} total profit: {}".format(stock_name, agent.get_total_profit()))
                print("--------------------------------")

def plot_action_profit(data, action_data, profit):
    plot.plot(range(len(data)), data)
    plt.xlabel("data")
    plt.ylabel("price")

    buy, sell = False, False
    for d in range(len(data) - 1):
        if action_data == 1: # buy
            buy, = plt.plot(d, data[d], 'g*')
        elif action_data == 2:
            sel, = plt.plot(d, data[d], 'r+')
    if buy and sell:
        plt.legend([buy, sell], ["Buy", "Sell"])
    plt.title("Total Profit: {}".format(profit))
    plt.savefig("buy_sell.png")
    plt.show()

if __name__ == "__main__":
    main()