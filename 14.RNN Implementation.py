import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


def get_params(vocab_size, num_hiddens, device):
    """
    初始化模型参数
    :param vocab_size: 词表的大小，在本节中为28：26个字母+空格+<UNK>
    :param num_hiddens: 隐藏层的参数数量
    :param device:  训练设备，一般为本机GPU
    :return:返回初始化后的模型参数
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):  # 根据指定的shape进行参数初始化
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

"""
循环神经网络模型
"""


def init_rnn_state(batch_size, num_hiddens, device):
    """
    循环神经网络中，该函数在初始化中返回隐状态
    :param batch_size:训练批量大小
    :param num_hiddens:隐藏层参数量
    :param device:训练设备
    :return:返回隐藏层的状态 H（批量大小，隐藏单元数）
    """
    return (torch.zeros((batch_size, num_hiddens), device=device), )


def rnn(inputs, state, params):
    """
    在一个时间步内计算隐状态和输出
    :param inputs:输入张量（时间步长，批量大小，词表大小）
    :param state:隐状态
    :param params:训练参数
    :return:输出(时间步长*批量大小，词表大小)，拼接而成  该步计算后的隐状态H
    """
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    """从零开始实现的循环神经网络模型，集成上面的函数"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """
    在prefix后面生成新字符
    :param prefix: 输入的字符串
    :param num_preds: 预测的长度限制
    :param net: 使用的网络
    :param vocab: 词表
    :param device: 训练设备
    :return: 预测生成的字符串
    """
    state = net.begin_state(batch_size=1, device=device)#预测单一字符串，所以批量大小为1
    outputs = [vocab[prefix[0]]]#先取出第一个char的index
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))# output的最后一个char取出作为输入
    for y in prefix[1:]:  # 预热期，主要用于生成隐藏层状态
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """
    训练网络一个迭代周期
    :param net: 训练网络
    :param train_iter: 训练数据迭代器
    :param loss: 损失值
    :param updater: 优化器
    :param device: 训练设备
    :param use_random_iter: 是否用了随机迭代（非随机迭代，一个batch中，前后是有相接的）
    :return: 困惑度
    """
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()#非随机抽样，前后有关联，但计算是在一次迭代中计算，故删去该轮前隐藏层H的计算图
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """
    训练模型
    :param net: 训练网络
    :param train_iter: 迭代器
    :param vocab: 词表
    :param lr: 学习率
    :param num_epochs:迭代次数
    :param device: 训练器
    :param use_random_iter:是否使用随机抽样
    :return: None
    """
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


if __name__ == '__main__':
    # 序列抽样
    num_hiddens = 512
    net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                          init_rnn_state, rnn)
    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
    # 随机抽样
    net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                          init_rnn_state, rnn)
    train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
              use_random_iter=True)