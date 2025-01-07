from Environment.env import Environment
from itertools import combinations
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import os
f_types = ["RE22"]
n = 5
for f_type in f_types:
    
    env = Environment(T = 100, domain_size = 1000, f_num = 2, function_type = f_type, seed = 0, noise_level=0.0)
    combinations_list = list(combinations(range(env.domain_dim), 2))

    np.random.shuffle(combinations_list)
    print(combinations_list[:min(n, len(combinations_list))])
    x_ticks = np.linspace(0, 1, 20)
    y_ticks = np.linspace(0, 1, 20)
    X = np.linspace(0, 1, 100)
    Y = np.linspace(0, 1, 100)

    fixed_points = [env.X[x] for x in np.random.choice(1000, size=5, replace=False)]
    for p, fixed_point in enumerate(fixed_points):
        cnt = 0
        fig = plt.figure(figsize=(20, 30))
        for i in range(min(n, len(combinations_list))):
            results = []
            for x in X:
                for y in Y:
                    point = fixed_point.copy()
                    point[combinations_list[i][0]] = x
                    point[combinations_list[i][1]] = y
                    results.append((x,y,env.F[0](point),env.F[1](point)))
            x, y, f1, f2 = zip(*results)
            F = [f1, f2]
            for j in range(2):
                ax = fig.add_subplot(n, 2, cnt + 1, projection='3d')
                cnt += 1
                scatter = ax.scatter(x, y, F[j], c=F[j], cmap='viridis')
                ax.set_xlabel(f'dim {combinations_list[i][0]}', labelpad=10)
                ax.set_ylabel(f'dim {combinations_list[i][1]}', labelpad=10)
                ax.set_title(f'{f_type} #{p} d {combinations_list[i]}', pad=20)
                ax.xaxis.set_major_locator(MaxNLocator(20))
                ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.xaxis.set_ticks(x_ticks)
                ax.xaxis.set_ticklabels([f'{tick:.3f}' for tick in x_ticks])
                ax.xaxis.set_tick_params(which='major', labelsize=8, rotation=45)

                ax.yaxis.set_major_locator(MaxNLocator(20))
                ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                ax.yaxis.set_ticks(y_ticks)
                ax.yaxis.set_ticklabels([f'{tick:.3f}' for tick in y_ticks])
                ax.yaxis.set_tick_params(which='major', labelsize=8, rotation=-20)

                plt.colorbar(scatter, ax=ax)
        os.makedirs(f'./Plot/{f_type}/', exist_ok=True)
        plt.savefig(f'./Plot/{f_type}/{f_type}_fix_point_{p}.png')
        plt.close()
