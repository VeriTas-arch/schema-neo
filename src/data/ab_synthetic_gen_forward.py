import torch
import numpy as np
import math
from torch.utils.data import Dataset


def get_item(item_index):
    sigma2 = 2
    size = 24
    allowed_values = [0, 1, 2, 3, 4, 5]

    if item_index not in allowed_values:
        raise ValueError("No such item index! Only six classes.")

    # 确定角度偏移
    miu = item_index * math.pi / 3

    # 生成等间隔点并归一化
    x = np.linspace(0, 2 * math.pi, size, endpoint=False)
    x_shifted = (x - miu + math.pi) % (2 * math.pi) - math.pi  # 归一化到 [-π, π]

    # 计算周期性高斯分布
    y = np.exp(-(x_shifted**2) / (2 * sigma2)) * 4

    return y


def get_order(order_index):
    rou = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    sigma2 = 0.05
    allowed_order = [1, 2, 3]
    y = np.exp(-((np.log(order_index) - np.log(rou)) ** 2) / (2 * sigma2)) * 2

    if order_index in allowed_order:
        return y
    else:
        raise ValueError("No such order index! only three classes.")


def get_data(
    stim_dur=4, num_train=20000, stim_123=None, class_index=None, silence_index=None
):
    stim_len = 200  # 在stim_3后面留一个dur
    train_x = np.zeros((num_train, stim_len, 25))
    train_y = np.zeros(num_train, np.int64)
    train_t = np.zeros((num_train, 5), np.int64)
    for i in range(len(train_y)):
        if stim_123 is None:
            # raise ValueError(stim_123)
            stim1 = 8
            stim2 = 24
            stim3 = 40
            stim4 = 30
            # stimr = np.random.choice(np.arange(stim2 + 2 * stim_dur, stim2 + 3 * stim_dur))
        else:
            stim1, stim2, stim3 = stim_123
        if class_index is None:
            class_i = np.random.choice(120)  # 实际范围是0~119
        else:
            class_i = class_index[i]

        item_1 = class_i // 20
        cue23 = class_i % 20
        cue2 = cue23 // 4
        cue3 = cue23 % 4
        p = [0, 1, 2, 3, 4, 5]
        p.remove(item_1)
        item_2 = p[cue2]
        p.remove(item_2)
        item_3 = p[cue3]

        # raise ValueError(item_3)

        stim_value_1 = get_item(item_1)
        stim_value_2 = get_item(item_2)
        stim_value_3 = get_item(item_3)

        if silence_index is not None:
            if silence_index == 1:
                stim_value_1 = np.zeros_like(stim_value_1)
            if silence_index == 2:
                stim_value_2 = np.zeros_like(stim_value_2)
            if silence_index == 3:
                stim_value_3 = np.zeros_like(stim_value_3)

        # raise ValueError(stim_value_3)
        # raise ValueError(stim3)

        train_x[i, stim1 : stim1 + stim_dur + 4, 0:24] = stim_value_1.reshape(1, 24)
        train_x[i, stim2 : stim2 + stim_dur + 4, 0:24] = stim_value_2.reshape(1, 24)
        train_x[i, stim3 : stim3 + stim_dur + 4, 0:24] = stim_value_3.reshape(1, 24)
        train_x[i, 0 : stim3 + stim_dur + stim4 + 4, 24:] = 1

        # raise ValueError(train_x1[i, stim1:stim1 + stim_dur, :])

        train_t[i, 0] = int(stim1)
        train_t[i, 1] = int(stim2)
        train_t[i, 2] = int(stim3)
        train_t[i, 3] = int(stim4)
        train_t[i, 4] = int(stim_len)

        train_y[i] = int(class_i)

    return train_x, train_y, train_t


def get_regress_target_long(class_i, train_t, stim_len):
    reg_out = np.zeros((stim_len, 7))
    t1, t2, t3, t4, t5 = train_t

    reg_out[:, :] = 0
    item_1 = class_i // 20
    cue23 = class_i % 20
    cue2 = cue23 // 4
    cue3 = cue23 % 4
    p = [0, 1, 2, 3, 4, 5]
    p.remove(item_1)
    item_2 = p[cue2]
    p.remove(item_2)
    item_3 = p[cue3]
    reg_out[t3 + t4 + 8 : t3 + t4 + 12 + 8, item_1] = 1
    reg_out[t3 + t4 + 12 + 8 : t3 + t4 + 12 + 12 + 8, item_2] = 1
    reg_out[t3 + t4 + 12 + 12 + 8 : t3 + t4 + 12 + 12 + 12 + 8, item_3] = 1

    reg_out[: t3 + t4 + 8, -1] = 1

    # reg_out[t1:t2, 6] = 1
    # reg_out[t2:t3, 7] = -1
    # reg_out[t3:, 8] = 1

    # raise ValueError(reg_out[0, :])

    return reg_out


def get_regress_target_wu(class_i, train_t, stim_len):
    # train_t, a list, length 3 [t1, t2, t3]
    reg_out = np.zeros((stim_len, 5))
    t1, t2, t3 = train_t

    reg_out[:t1, -1] = 1
    reg_out[t2:t3, class_i] = 1.0

    if class_i in [0, 1]:
        reg_out[t1:t2, 0:2] = 1.0
    if class_i in [2, 3]:
        reg_out[t1:t2, 2:4] = 1.0
    return reg_out


class forwardDataset(Dataset):
    def __init__(
        self,
        noise_sigma=0.05,
        stim_dur=4,
        num_data=2000,
        stim_123=None,
        class_index=None,
        regress_target_mode="hierarchy_long",
        silence_index=None,
    ):

        self.stim_len = 24 * 4 + 20
        self.stim_dur = stim_dur

        self.data_x, self.data_y, self.data_t = get_data(
            stim_dur=stim_dur,
            num_train=num_data,
            stim_123=stim_123,
            class_index=class_index,
            silence_index=silence_index,
        )
        self.noise_sigma = noise_sigma
        self.regress_target_mode = regress_target_mode

        # raise ValueError(self.data_x1)

        # pdb.set_trace()

    def __getitem__(self, index):
        data_x_i = self.data_x[index].copy()
        data_y_i = self.data_y[index].copy()
        data_t_i = self.data_t[index].copy()

        # raise ValueError(data_x1_i)

        if self.regress_target_mode == "hierarchy_wu":
            data_reg_i = get_regress_target_wu(data_y_i, data_t_i, self.stim_len)
            single_reg_i = np.zeros((3, 5))
            for i in range(3):
                single_reg_i[i, :] = data_reg_i[data_t_i[i] - 1, :]

        elif self.regress_target_mode == "hierarchy_long":
            data_reg_i = get_regress_target_long(data_y_i, data_t_i, self.stim_len)
            # single_reg_i = np.zeros((self.stim_len, 3))

        elif self.regress_target_mode == "constant":
            data_reg_i = np.zeros((data_x_i.shape[0], 10))
            data_reg_i[:, data_y_i] = 1.0
            single_reg_i = np.zeros((self.stim_len, 3))

        if self.noise_sigma > 0:
            # data_x_i[0:8,0:24] += self.noise_sigma * np.random.normal(0, 1, size=data_x_i[0:8,0:24].shape)
            # data_x_i[24:32, 0:24] += self.noise_sigma * np.random.normal(0, 1, size=data_x_i[24:32, 0:24].shape)
            # data_x_i[40:48, 0:24] += self.noise_sigma * np.random.normal(0, 1, size=data_x_i[40:48, 0:24].shape)
            data_x_i[48:, 0:24] += self.noise_sigma * np.random.normal(
                0, 1, size=data_x_i[48:, 0:24].shape
            )

        # (T,3)
        return (
            torch.FloatTensor(data_x_i),
            torch.LongTensor([data_y_i]),
            torch.LongTensor(data_t_i),
            torch.FloatTensor(data_reg_i),
        )

    def __len__(self):
        return len(self.data_y)

    def get_timepoints(self):
        actual_stim_dur = self.stim_dur + 4
        target_len = 12

        # stim1 = 8
        # stim2 = 24
        # stim3 = 40
        # stim4 = 30  # delay_len
        # stim5 = stim_len = 200
        data_t_i = self.data_t[0].copy()

        stim1_start = data_t_i[0]
        stim2_start = data_t_i[1]
        stim3_start = data_t_i[2]

        stim1_off = stim1_start + actual_stim_dur
        stim2_off = stim2_start + actual_stim_dur
        stim3_off = stim3_start + actual_stim_dur

        stim1_end = stim2_start
        stim2_end = stim3_start
        stim3_end = stim3_off

        delay_start = stim3_end
        delay_end = delay_start + data_t_i[3]

        target1_start = delay_end
        target1_end = target1_start + target_len
        target2_start = target1_end
        target2_end = target2_start + target_len
        target3_start = target2_end
        target3_end = target3_start + target_len

        timepoints = dict(
            stim1_start=stim1_start,
            stim1_off=stim1_off,
            stim1_end=stim1_end,
            stim2_start=stim2_start,
            stim2_off=stim2_off,
            stim2_end=stim2_end,
            stim3_start=stim3_start,
            stim3_off=stim3_off,
            stim3_end=stim3_end,
            delay_start=delay_start,
            delay_end=delay_end,
            target1_start=target1_start,
            target1_end=target1_end,
            target2_start=target2_start,
            target2_end=target2_end,
            target3_start=target3_start,
            target3_end=target3_end,
            end=data_t_i[4],
        )

        return timepoints
