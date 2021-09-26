import numpy as np
import matplotlib.pyplot as plt




if __name__ == '__main__':
    fig, ax = plt.subplots(2, 3)
    t_r = range(1,100)
    t_g_r = t_r
    e_t = t_r
    expend = t_r
    ax[0][0].cla()
    ax[0][1].cla()
    ax[0][2].cla()
    ax[1][0].cla()
    ax[0][0].plot(t_r, label='ToTal_Reward')
    ax[0][0].set_title('asd')
    ax[0][0].set_xlabel('x')
    ax[0][0].set_ylabel('y')
    # 设置下标上下限
    ax[0][0].set_xlim(0,9)
    #指定下标
    ax[0][0].set_xticks([1,2,3,4,5,6])
    ax[0][1].axis("off")
    ax[0][1].plot(t_g_r, label='ToTal_Reward2')
    ax[0][2].plot(e_t, label='ToTal_Reward3')
    # ax[0][2].set_spines['right'].set_color('none')
    # ax[0][2].set_spines['top'].set_color('none')
    # ax[0][2].set_xaxis.set_ticks_position('bottom')
    # ax[0][2].set_spines['bottom'].set_position(('data', 0))
    # ax[0][2].set_yaxis.set_ticks_position('left')
    # ax[0][2].set_spines['left'].set_position(('data', 0))
    ax[1][0].plot(expend, label='ToTal_Reward3')
    # plt.title("Matplotlib demo")
    # plt.show()
    plt.pause(1)

    plt.show()
    pass
