import os
import torch
from d2l import torch as d2l



d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                          '94646ad1522d915e7b0f9296181140edcf86a4f5')


def read_data_nmt():
    """
    载入“英语－法语”数据集
    :return: 返回读入的英语-数据集法语   形式如下
    Go.         Va!
    Run!        Courez!
    Wow!        Ca  alors!
    """
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read()


#@save
def preprocess_nmt(text):
    """
    预处理“英语－法语”数据集
    :param text: 读取的数据集
    :return: 空格代替不间断空格，小写化，在单词和标点符号之间插入空格的 预处理后的数据集   形式如下
    go .        va !
    hi .        salut !
    run !       cours !
    run !       courez !
    who ?       qui ?
    wow !       ça alors !
    """
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """
    词元化“英语－法语”数据数据集
    :param text: 预处理后的数据集
    :param num_examples: 是否指定处理数据集的前num_examples个数据
    :return:词元化后的数据集
    """
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        print(f"i:{i} {line}")
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


def truncate_pad(line, num_steps, padding_token):
    """
    截断或填充文本序列
    :param line: 读入行数
    :param num_steps: 处理长度
    :param padding_token: 填充的字符
    :return: 返回规范化长度的数据
    """
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def build_array_nmt(lines, vocab, num_steps):
    """
    将机器翻译的文本序列转换成小批量
    :param lines: 源数据
    :param vocab: 词表
    :param num_steps:处理步长
    :return: 返回训练的X，Y 以及有效长度（填充前的长度）
    """
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


def load_data_nmt(batch_size, num_steps, num_examples=600):
    """
    返回翻译数据集的迭代器和词表
    :param batch_size: 训练批量大小
    :param num_steps: 训练步长
    :param num_examples: 训练num_examples个
    :return: 迭代器， 英语词表， 法语词表
    """
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


if __name__ == '__main__':
    # 读取数据集
    raw_text = read_data_nmt()
    print(raw_text[:75])
    # 预处理
    text = preprocess_nmt(raw_text)
    print(text[:80])
    # 词元化
    source, target = tokenize_nmt(text)
    source[:6], target[:6]
    # 制成词表
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # 读出小批量数据
    train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
    for X, X_valid_len, Y, Y_valid_len in train_iter:
        print('X:', X.type(torch.int32))
        print('X的有效长度:', X_valid_len)
        print('Y:', Y.type(torch.int32))
        print('Y的有效长度:', Y_valid_len)
        break