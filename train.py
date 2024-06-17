import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os

import net
import mtsp
from scipy.stats import ttest_rel

if __name__ == '__main__':
    DEVICE = net.DEVICE
    net1 = net.ActNet()
    net1 = net1.to(DEVICE)
    net2 = net.ActNet()
    net2 = net2.to(DEVICE)
    # net1.load_state_dict(torch.load('/home/ballade/Desktop/Project/MTSP/mywork/save/epoch8-i99-dis_3.88936.pt'))

    epochs = 10
    times = 200#mini batch
    batch = net.batch
    city_size = net.citySize

    bl_alpha = 0.05  # 做t-检验更新baseline时所设置的阈值
    test2save_times = 20  # 训练过程中每次保存模型所需的测试batch数
    min = 100  # 当前已保存的所有模型中测试路径长度的最小值

    #用mtsp生成数据
    # X = torch.rand(batch*times,city_size,2)
    # tX = torch.rand(batch*test2save_times,city_size,2)
    n_agent = net.agentSize
    n_cities = city_size - n_agent
    env = mtsp.mtsp(n_cities, n_agent, batch, times, test2save_times, 2024)

    save_dir = '/home/ballade/Desktop/Project/MTSP/mywork/save/'

    #训练
    opt = optim.Adam(net1.parameters(), lr=0.0005)
    # torch.autograd.set_detect_anomaly(True)

    for epoch in range(epochs):
        for i in range(times):
            torch.cuda.empty_cache()
            # x = X[i*batch:(i+1)*batch].to(DEVICE)
            x = env.get_batch(i, True).to(DEVICE)

            time1 = time.time()
            seq2,pro2,dis2 = net2(x,is_train=False)#baseline
            seq1,pro1,dis1 = net1(x,is_train=True)
            time2 = time.time()
            # print('time:',time2-time1)

            pro = torch.log(pro1)#(batch, city_size)
            loss = torch.sum(pro, dim=1)#(batch)
            score = dis1-dis2
            score = score.detach()#(batch)
            # loss = torch.mean(loss*score)
            loss = score * loss
            loss = torch.sum(loss)/batch

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net1.parameters(), 1)#梯度裁剪,防止梯度爆炸
            opt.step()
            print('epoch={},i={},mean_dis1={},mean_dis2={}'.format(epoch, i, torch.mean(dis1), torch.mean(
                dis2)))  # ,'disloss:',t.mean((dis1-dis2)*(dis1-dis2)), t.mean(t.abs(dis1-dis2)), nan)

            # OneSidedPairedTTest(做t-检验看当前Sampling的解效果是否显著好于greedy的解效果,如果是则更新使用greedy策略作为baseline的net2参数)
            if (dis1.mean() - dis2.mean()) < 0:
                tt, pp = ttest_rel(dis1.cpu().numpy(), dis2.cpu().numpy())
                p_val = pp / 2
                assert tt < 0, "T-statistic should be negative"
                if p_val < bl_alpha:
                    print('Update baseline')
                    net2.load_state_dict(net1.state_dict())

            # 每隔xxx步做测试判断结果有没有改进，如果改进了则把当前模型保存下来
            if (i + 1) % 100 == 0:
                length = torch.zeros(1).to(DEVICE)
                for j in range(test2save_times):
                    torch.cuda.empty_cache()
                    # t_x = tX[j * batch: (j + 1) * batch].to(DEVICE)
                    t_x = env.get_batch(j, False).to(DEVICE)
                    seq, pro, dis = net1(t_x, is_train=False)
                    length = length + torch.mean(dis)
                length = length / test2save_times
                if length < min:
                    torch.save(net1.state_dict(), os.path.join(save_dir,'date{}-epoch{}-i{}-dis_{:.5f}.pt'.format(
                        time.strftime("%Y-%m-%d", time.localtime()), epoch, i, length.item())))
                    # torch.save(net1.state_dict(), os.path.join(save_dir,
                    #                                        'epoch{}-i{}-dis_{:.5f}.pt'.format(
                    #                                            epoch, i, length.item())))
                    min = length
                print('i=',i,'min=', min.item(), 'length=', length.item())
                # print('epoch={},i={},mean_dis1={},mean_dis2={}'.format(epoch, i, torch.mean(dis1), torch.mean(dis2)))


            