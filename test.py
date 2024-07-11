import mtsp
import greedy_allocation_lib as gal
import ga_allocation_lib as ga
import net
import torch
import time

if __name__ == '__main__':
    n_agent = net.agentSize
    n_cities =net.nodeSize - n_agent
    n_batch = 512
    n_times = 1
    env = mtsp.mtsp(n_cities, n_agent, n_batch, n_times,n_times,2025)

###only dispose the situation of batch_size = 1
    # planner = gal.GreedyTaskAllocationPlanner()
    # x = env.get_batch(0, False)
    # agent_poses = x[0, 0:n_agent, :]
    # task_poses = x[0, n_agent:, :]
    # time1 = time.time()
    # schedules = planner.greedy_allocate(agent_poses, task_poses)
    # time2 = time.time()
    # print('time:', time2-time1)
    # dis_gal = planner.allocation_distance_eval(agent_poses, task_poses, schedules)
    # # dis_test = env.get_distance(schedules, 0, True)
    # print(schedules)
    # print(dis_gal)
    # # print(dis_test)
    # env.render(schedules, 0, True)

    # planner_ga = ga.GATaskAllocationPlanner()
    # time1 = time.time()
    # schedules_ga = planner_ga.ga_allocate(planner_ga.dataStructTransform(agent_poses), planner_ga.dataStructTransform(task_poses))
    # time2 = time.time()
    # print('time:', time2-time1)
    # dis_ga = planner.allocation_distance_eval(agent_poses, task_poses, schedules_ga)
    # # dis_test = env.get_distance(schedules_ga, 0, True)
    # print(schedules_ga)
    # print(dis_ga)
    # # print(dis_test)
    # env.render(schedules_ga, 0, True)

    # net0 = net.ActNet()
    # net0 = net0.to(net.DEVICE)
    # net0.load_state_dict(torch.load('/home/ballade/Desktop/Project/MTSP/mywork/save/date2024-06-17-epoch8-i199-dis_3.35644.pt'))
    # x = x.to(net.DEVICE)
    # time1 = time.time()
    # seq, pro, dis = net0(x, is_train=False)
    # time2 = time.time()
    # print('time:', time2-time1)
    # seq = seq.view(-1).cpu().numpy().astype(int)
    # dis = dis.cpu().numpy()
    # # dis_test = env.get_distance(seq, 0)
    # print(seq)
    # print(dis)
    # # print(dis_test)
    # env.render(seq, 0)
###

    x = env.get_batch(0, False)

    planner = gal.GreedyTaskAllocationPlanner()
    time_gal = 0
    dis_gal = 0
    for i in range(n_batch):
        agent_poses = x[i, 0:n_agent, :]
        task_poses = x[i, n_agent:, :]
        time1 = time.time()
        schedules = planner.greedy_allocate(agent_poses, task_poses)
        time2 = time.time()
        time_gal += time2 - time1
        dis_gal += planner.allocation_distance_eval(agent_poses, task_poses, schedules)
    time_gal /= n_batch
    dis_gal /= n_batch
    print('time_gal:', time_gal)
    print('dis_gal:', dis_gal)

    # planner_ga = ga.GATaskAllocationPlanner()
    # time_ga = 0
    # dis_ga = 0
    # for i in range(n_batch):
    #     agent_poses = x[i, 0:n_agent, :]
    #     task_poses = x[i, n_agent:, :]
    #     time1 = time.time()
    #     schedules_ga = planner_ga.ga_allocate(planner_ga.dataStructTransform(agent_poses), planner_ga.dataStructTransform(task_poses))
    #     time2 = time.time()
    #     time_ga += time2 - time1
    #     dis_ga += planner.allocation_distance_eval(agent_poses, task_poses, schedules_ga)
    # time_ga /= n_batch
    # dis_ga /= n_batch
    # print('time_ga:', time_ga)
    # print('dis_ga:', dis_ga)
        
    net0 = net.ActNet()
    net0 = net0.to(net.DEVICE)
    net0.load_state_dict(torch.load('/home/ballade/Desktop/Project/MTSP/mywork/save/date2024-06-17-epoch8-i199-dis_3.35644.pt'))
    x = x.to(net.DEVICE)
    time1 = time.time()
    seq, pro, dis = net0(x, is_train=False)
    time2 = time.time()
    time_net = time2 - time1
    dis_net = torch.mean(dis).cpu().numpy()
    print('time_net:', time_net)
    print('dis_net:', dis_net)




    
    