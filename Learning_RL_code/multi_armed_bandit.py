# 多臂老虎机问题
import numpy as np
import matplotlib.pyplot as plt

class BernoulliBandit:
    '''
    伯努利多臂老虎机，输入K表示拉杆的个数
    '''
    def __init__(self, K):
        # 生成一个K个元素(均在0~1之间)的数组
        self.probs = np.random.uniform(size=K) # 随机生成K个0～1的数,作为拉动每根拉杆的获奖
        self.K = K
        self.best_idx = np.argmax(self.probs) # 返回最大值的索引
        self.best_prob = self.probs[self.best_idx] # 得到最大的概率

    def step(self, k): # 玩家选择第k个摇杆时，是否会获奖
        if np.random.rand() < self.probs[k]: # 小于才会以self.probs[k]的概率返回1
            return 1
        else:
            return 0

np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print("随机生成了一个%d臂伯努利老虎机" % K)
print("获奖概率最大的拉杆为%d号,其获奖概率为%.4f" %
      (bandit_10_arm.best_idx, bandit_10_arm.best_prob))


class Slover:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K) # 每根拉杆被拉动的次数，通过一个数组记录
        self.regret = 0 # 当前步骤的累计regret值
        self.actions = [] # 记录每一步的动作
        self.regrets = [] # 记录每一步的累计regret

    def update_regret(self, k):
        # regret 的定义是期望奖励最高的动作的奖励 - 当前动作的期望奖励; 由于是伯努利分布，返回的奖励是0，1; 所以每个动作的期望就是获得奖励的概率
        self.regret += self.bandit.best_prob - self.bandit.probs[k] # 更新累计regret
        self.regrets.append(self.regret) # 记录累计regret

    def run_one_step(self):
        raise NotImplementedError # 此方法在这个类中没有实现，可能会在子类中实现，复写

    def run(self, num_steps):
        # 运行num_steps次实验
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)

# 拉动拉杆的策略
class EpsilonGreedy(Slover):
    '''
    epsilon贪心算法，继承slover类
    '''
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * bandit.K) # 每个拉杆的期望的估计值，初始化为1

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K) # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates) # 选取以往估计中期望最高的动作
        r = self.bandit.step(k) # 得到此次动作的奖励
        self.estimates[k] += 1. /(self.counts[k] + 1) * (r - self.estimates[k]) # 更新期望值
        return k

def plot_results(solvers, solver_names):
    '''
    生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称
    '''
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label = solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    try:
        plt.show()
    except Exception:
        plt.savefig('cumulative_regrets.png', dpi=150)
    finally:
        plt.close()

np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])

