import mtsp
import greedy_allocation_lib as gal
import net
import torch
import time

if __name__ == '__main__':
    n_agent = 1
    n_cities =50
    env = mtsp.mtsp(n_cities, n_agent, 1, 1,1,2025)

    planner = gal.GreedyTaskAllocationPlanner()
    x = env.get_batch(0, False)
    agent_poses = x[0, 0:n_agent, :]
    task_poses = x[0, n_agent:, :]
    time1 = time.time()
    schedules = planner.greedy_allocate(agent_poses, task_poses)
    time2 = time.time()
    print('time:', time2-time1)
    dis_gal = planner.allocation_distance_eval(agent_poses, task_poses, schedules)
    dis_test = env.get_distance(schedules, 0)
    print(schedules)
    print(dis_gal)
    print(dis_test)
    schedules = torch.tensor(schedules)
    schedules = schedules.view(n_agent, n_cities)
    env.render(schedules, 0)

    net0 = net.ActNet()
    net0 = net0.to(net.DEVICE)
    net0.load_state_dict(torch.load('/home/ballade/Desktop/Project/MTSP/mywork/save/date2024-06-14-epoch0-i99-dis_6.52384.pt'))
    x = x.to(net.DEVICE)
    time1 = time.time()
    seq, pro, dis = net0(x, is_train=False)
    time2 = time.time()
    print('time:', time2-time1)
    seq = seq.view(n_agent, n_cities).cpu().numpy().astype(int)
    seq = seq-1
    dis = dis.cpu().numpy()
    dis_test = env.get_distance(seq, 0)
    print(seq)
    print(dis)
    print(dis_test)
    env.render(seq, 0)

    
    # agent_poses = env.start
    # task_poses = env.cities
    # schedules = planner.greedy_allocate(agent_poses, task_poses)
    # print(schedules)
    # env.tour = schedules
    # agent_distances = planner.allocation_distance_eval(agent_poses, task_poses, schedules)
    # print(agent_distances)
    # env.render()
    