import torch
from torch import nn


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    gpu_num = torch.cuda.device_count()
    gpu_device_id = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(gpu_device_id)
    print(use_gpu, gpu_num, gpu_device_id, gpu_name)
    # 转Tensor的时候要 换GPU
    tensor1 = torch.tensor([1,2,3])
    print(tensor1, tensor1.device)
    tensor1 = tensor1.cuda(0)

    print(tensor1, tensor1.device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = tensor1.to(device)
    tensor1 = torch.tensor([1, 2, 3], device=device, dtype=torch.float32)

    # gpu的net 要输入GPU的数据，不然报错
    net = nn.Linear(3, 1)
    print(type(net.parameters()))
    print(list(net.parameters())[0].device)
    net = net.to(device)
    print(list(net.parameters())[0].device)
    print(net(tensor1))

    # 把数据，网络，与损失函数转换到GPU上
    # model = get_model()
    # loss_f = t.nn.CrossEntropyLoss()
    # if (use_gpu):
    #     model = model.cuda()
    #     loss_f = loss_f.cuda()
    pass