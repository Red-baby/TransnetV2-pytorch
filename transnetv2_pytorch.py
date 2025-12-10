# 下面为原文件增加了逐行/逐段中文注释，面向初学者讲解设计要点与数据流。
# 注：未修改程序逻辑，仅添加注释，便于课堂讲解或自学阅读。

import torch
import torch.nn as nn
import torch.nn.functional as functional

import random

# 顶层模型类：TransNetV2
# 这是 PyTorch 的实现，整体思路与 TF 版本类似：
# - 输入为 uint8 视频帧序列 [B, T, H, W, C]
# - 经过一系列 3D 卷积块（时间 + 空间）提取时空特征
# - 对空间维度做池化或展平，得到每帧的向量表示
# - 可选地拼接帧相似度/颜色直方图等辅助特征
# - 全连接层输出每帧的打分（one_hot），可选多热输出
class TransNetV2(nn.Module):

    def __init__(self,
                 F=16, L=3, S=2, D=1024,
                 use_many_hot_targets=True,
                 use_frame_similarity=True,
                 use_color_histograms=True,
                 use_mean_pooling=False,
                 dropout_rate=0.5,
                 use_convex_comb_reg=False,  # not supported
                 use_resnet_features=False,  # not supported
                 use_resnet_like_top=False,  # not supported
                 frame_similarity_on_last_layer=False):  # not supported
        super(TransNetV2, self).__init__()

        # 参数解释（常用超参）
        # F: 初始卷积 filters 数量（基础通道数）
        # L: 堆叠的 block 数（层级深度）
        # S: 每个 block 内的 DilatedDCNNV2 子块数
        # D: 全连接层隐藏维度
        # use_*: 是否启用若干可选模块（帧相似度、颜色直方图等）

        # 以下未实现的选项直接抛出异常，提醒使用者
        if use_resnet_features or use_resnet_like_top or use_convex_comb_reg or frame_similarity_on_last_layer:
            raise NotImplemented("Some options not implemented in Pytorch version of Transnet!")

        # 构建一系列 StackedDDCNNV2 模块（类似 ResNet 的 stage）
        # 第一个 block 输入通道为 3（RGB），后续 block 的输入通道由上一 block 的输出通道决定
        # 这里每个 DilatedDCNNV2 会输出 filters*4 个通道（四路并行膨胀卷积 concat）
        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.)] +
            [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i) for i in range(1, L)]
        )

        # 可选的辅助特征层：FrameSimilarity 将若干 block 的时空特征转为局部相似度描述符
        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2 ** i) * 4 for i in range(L)]), lookup_window=101, output_dim=128, similarity_dim=128, use_bias=True
        ) if use_frame_similarity else None

        # 可选的颜色直方图层：直接从原始 uint8 帧计算直方图并生成相似度描述
        self.color_hist_layer = ColorHistograms(
            lookup_window=101, output_dim=128
        ) if use_color_histograms else None

        # Dropout 层（如果提供 dropout_rate，则用于训练时的正则）
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate is not None else None

        # 计算 fc 第一层的输入维度（基于最后一个 block 的输出通道与空间尺寸）
        # 这里写死了空间尺寸 3x6（与数据预处理/配置对应）
        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6  # 3x6 for spatial dimensions
        if use_frame_similarity: output_dim += 128
        if use_color_histograms: output_dim += 128

        # 全连接层与分类头
        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        # 当使用 many_hot 目标时，额外的 second head
        self.cls_layer2 = nn.Linear(D, 1) if use_many_hot_targets else None

        self.use_mean_pooling = use_mean_pooling
        # 将模型置为 eval 模式作为默认状态（用户可在训练时调用 train()）
        self.eval()

    # forward (PyTorch 风格)：
    # 输入断言，要求 uint8 且空间尺寸为 [27,48,3]（项目设定）
    def forward(self, inputs):
        assert isinstance(inputs, torch.Tensor) and list(inputs.shape[2:]) == [27, 48, 3] and inputs.dtype == torch.uint8, \
            "incorrect input type and/or shape"
        # 将输入从 [B, T, H, W, 3] -> [B, 3, T, H, W] 并转 float（0..255 -> 0..1）
        x = inputs.permute([0, 4, 1, 2, 3]).float()
        x = x.div_(255.)

        # 逐个通过 block（stage），收集每个 block 的输出（用于 frame similarity）
        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        # 此处 x 为最后一个 block 的输出，形状 [B, C, T, H, W]

        # 两种空间聚合策略：
        # - use_mean_pooling=True：对空间维取均值 -> [B, C, T] -> 转置为 [B, T, C]
        # - 否则：把 [B, C, T, H, W] 调整为 [B, T, H, W, C] 然后展平成向量 [B, T, H*W*C]
        if self.use_mean_pooling:
            x = torch.mean(x, dim=[3, 4])   # 对 H,W 求均值
            x = x.permute(0, 2, 1)          # [B, C, T] -> [B, T, C]
        else:
            # 先把通道移到最后，便于按帧展平
            x = x.permute(0, 2, 3, 4, 1)    # [B, C, T, H, W] -> [B, T, H, W, C]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, T, H*W*C]

        # 若启用了帧相似度层，把它计算出的描述拼接到每帧特征的前面（按通道方向）
        if self.frame_sim_layer is not None:
            # frame_sim_layer 接受 block_features（list of tensors），返回 [B, T, feat_dim]
            x = torch.cat([self.frame_sim_layer(block_features), x], 2)

        # 若启用了颜色直方图层，从原始 uint8 帧计算颜色相似度描述并拼接
        if self.color_hist_layer is not None:
            # color_hist_layer 接受原始 frames（uint8），返回 [B, T, feat_dim]
            x = torch.cat([self.color_hist_layer(inputs), x], 2)

        # 全连接 + ReLU
        x = self.fc1(x)
        x = functional.relu(x)

        # dropout（训练时生效）
        if self.dropout is not None:
            x = self.dropout(x)

        # 输出单帧分数
        one_hot = self.cls_layer1(x)

        # 如果多热目标需要第二个 head，则返回 (one_hot, {many_hot: logits})
        if self.cls_layer2 is not None:
            return one_hot, {"many_hot": self.cls_layer2(x)}

        return one_hot


# ------------------------
# StackedDDCNNV2：一个 stage，包含若干个 DilatedDCNNV2，然后做 shortcut + pool
# 设计思想：
# - 每个 DilatedDCNNV2 会把输入映射为 (filters*4) 个通道（4 路膨胀卷积 concat）
# - 当多个子块串联时，第一子块的输入通道为 in_filters，之后为 filters*4（上一层输出）
# - shortcut 提供残差连接；pool 降低空间分辨率（pool kernel (1,2,2)）
class StackedDDCNNV2(nn.Module):

    def __init__(self,
                 in_filters,
                 n_blocks,
                 filters,
                 shortcut=True,
                 use_octave_conv=False,  # not supported
                 pool_type="avg",
                 stochastic_depth_drop_prob=0.0):
        super(StackedDDCNNV2, self).__init__()

        # Octave convolution 在 PyTorch 版本中未实现
        if use_octave_conv:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")

        # 仅支持 max 或 avg 池化
        assert pool_type == "max" or pool_type == "avg"
        if use_octave_conv and pool_type == "max":
            print("WARN: Octave convolution was designed with average pooling, not max pooling.")

        self.shortcut = shortcut
        # 逐个创建 n_blocks 个 DilatedDCNNV2
        # 注意 in_filters 传递规则：第一个 block 使用传入的 in_filters，其它使用 filters*4（上一 block 输出）
        self.DDCNN = nn.ModuleList([
            DilatedDCNNV2(in_filters if i == 1 else filters * 4, filters, octave_conv=use_octave_conv,
                          activation=functional.relu if i != n_blocks else None) for i in range(1, n_blocks + 1)
        ])
        # 池化层：时间维保持，空间维下采样为一半
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        # 随机深度概率（训练时随机跳过子层，用于正则化）
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    # forward：串联子块并在最后应用 shortcut + 激活 + 池化
    def forward(self, inputs):
        x = inputs
        shortcut = None

        # 依次通过所有 DilatedDCNNV2
        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                # 保存第一个子块的输出作为 shortcut（与原实现一致）
                shortcut = x

        # 把并联后的输出做激活（如果子块最后一个没有激活，则这里统一激活）
        x = functional.relu(x)

        # shortcut 处理（包含随机深度逻辑）
        if self.shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.:
                # 训练时按概率跳过（直接使用 shortcut），否则相加；推理时做缩放保持期望
                if self.training:
                    if random.random() < self.stochastic_depth_drop_prob:
                        x = shortcut
                    else:
                        x = x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                # 无随机深度，直接残差相加，避免原地修改以免 autograd 版本冲突
                x = x + shortcut

        # 空间下采样（保留时间维）
        x = self.pool(x)
        return x


# ------------------------
# DilatedDCNNV2：并行多尺度膨胀卷积的实现（dilation rates = 1,2,4,8）
# 目的是同时捕获不同的时间范围的特征（时间维的 dilations）
class DilatedDCNNV2(nn.Module):

    def __init__(self,
                 in_filters,
                 filters,
                 batch_norm=True,
                 activation=None,
                 octave_conv=False):  # not supported
        super(DilatedDCNNV2, self).__init__()

        if octave_conv:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")

        assert not (octave_conv and batch_norm)

        # 四路并行 Conv3D（不同的时间向膨胀率），每路输出 filters 个通道
        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)

        # 若启用了 batch_norm，则在 concat 后对 filters*4 通道做 BN
        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        # activation 可以传入 functional.relu 或 None（最后一个子块常设为 None，交由上层统一激活）
        self.activation = activation

    def forward(self, inputs):
        # 每路卷积独立作用于同一输入
        conv1 = self.Conv3D_1(inputs)
        conv2 = self.Conv3D_2(inputs)
        conv3 = self.Conv3D_4(inputs)
        conv4 = self.Conv3D_8(inputs)

        # 沿通道维度拼接（channels-first: dim=1）
        x = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        # BN + 激活（如设置）
        if self.bn is not None:
            x = self.bn(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


# ------------------------
# Conv3DConfigurable：可配置的 Conv3D，支持 separable (2+1)D 或者 标准 3D
# separable: 先做 1x3x3 空间卷积，再做 3x1x1 时间卷积（2+1D）
class Conv3DConfigurable(nn.Module):

    def __init__(self,
                 in_filters,
                 filters,
                 dilation_rate,
                 separable=True,
                 octave=False,  # not supported
                 use_bias=True,
                 kernel_initializer=None):  # not supported
        super(Conv3DConfigurable, self).__init__()

        if octave:
            raise NotImplemented("Octave convolution not implemented in Pytorch version of Transnet!")
        if kernel_initializer is not None:
            raise NotImplemented("Kernel initializers are not implemented in Pytorch version of Transnet!")

        assert not (separable and octave)

        # 当 separable=True 时，使用两层实现 (2+1)D 卷积：先空间再时间
        if separable:
            # conv1: 输入 -> 2*filters，kernel=(1,3,3)，只对空间做感受野
            conv1 = nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3),
                              dilation=(1, 1, 1), padding=(0, 1, 1), bias=False)
            # conv2: 2*filters -> filters，kernel=(3,1,1)，在时间维使用 dilation_rate
            conv2 = nn.Conv3d(2 * filters, filters, kernel_size=(3, 1, 1),
                              dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 0, 0), bias=use_bias)
            self.layers = nn.ModuleList([conv1, conv2])
        else:
            # 标准 3D 卷积，kernel=3，时间维使用 dilation_rate，空间维 padding=1
            conv = nn.Conv3d(in_filters, filters, kernel_size=3,
                             dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 1, 1), bias=use_bias)
            self.layers = nn.ModuleList([conv])

    def forward(self, inputs):
        # 顺序执行 layers（对 separable 情形会先空间卷积再时间卷积）
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


# ------------------------
# FrameSimilarity：计算局部帧相似度描述
# 输入：一个 block_features 的 list（每项是 [B, C, T, H, W]）
# 流程：
# - 对每个 block 做全空间平均 -> 得到 per-frame 向量 [B, C, T]
# - concat 所有 block 的通道 -> [B, total_C, T] -> 转置为 [B, T, total_C]
# - 投影到 similarity_dim 并 L2 归一化，再计算帧间相似矩阵 [B, T, T]
# - 对每帧取 lookup_window 局部相似行，FC 映射到 output_dim
class FrameSimilarity(nn.Module):

    def __init__(self,
                 in_filters,
                 similarity_dim=128,
                 lookup_window=101,
                 output_dim=128,
                 stop_gradient=False,  # not supported
                 use_bias=False):
        super(FrameSimilarity, self).__init__()

        if stop_gradient:
            raise NotImplemented("Stop gradient not implemented in Pytorch version of Transnet!")

        # 投影层把每帧向量投影到 similarity_dim，用于计算余弦相似度
        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        # 最后一个全连接把 lookup_window 个相似度值映射到 output_dim
        self.fc = nn.Linear(lookup_window, output_dim)

        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    def forward(self, inputs):
        # inputs 是 list：对每个 block 做 mean([H,W])，得到 [B, C, T]
        x = torch.cat([torch.mean(x, dim=[3, 4]) for x in inputs], dim=1)
        # 转置为 [B, T, total_C]，方便 batch 矩阵乘法计算相似度
        x = torch.transpose(x, 1, 2)

        # 投影到 similarity_dim 并做 L2 归一化（用于余弦相似度）
        x = self.projection(x)
        x = functional.normalize(x, p=2, dim=2)

        batch_size, time_window = x.shape[0], x.shape[1]
        # 计算相似度矩阵 [B, T, T]
        similarities = torch.bmm(x, x.transpose(1, 2))
        # 为了对窗口提取方便，在两侧 pad (lookup_window-1)/2
        similarities_padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        # 构建用于索引的三维索引矩阵：batch_indices, time_indices, lookup_indices
        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]) + time_indices

        # 从 padded 矩阵中取出局部窗口 [B, T, lookup_window]
        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]
        # 最后通过 fc 并 ReLU
        return functional.relu(self.fc(similarities))


# ------------------------
# ColorHistograms：从 uint8 帧计算 512-bin 颜色直方图并基于直方图计算相似度窗口
# 实现要点：
# - 把 RGB 8bit 各取高 3 位 -> 9-bit 索引（0..511）
# - 对每帧的像素计数（scatter_add），归一化后得到每帧 512-d 向量
# - 计算帧间相似度（dot product），再取局部 lookup_window 并通过可选 fc 映射
class ColorHistograms(nn.Module):

    def __init__(self,
                 lookup_window=101,
                 output_dim=None):
        super(ColorHistograms, self).__init__()

        # 若指定 output_dim 则最后用一个线性层降维
        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window
        assert lookup_window % 2 == 1, "`lookup_window` must be odd integer"

    @staticmethod
    def compute_color_histograms(frames):
        # frames: uint8 tensor [B, T, H, W, 3]
        frames = frames.int()  # 转为 int 便于位运算

        def get_bin(frames):
            # 计算 0..511 的 bin 值：每通道右移 5 位（保留高 3 位）然后拼接
            R, G, B = frames[:, :, 0], frames[:, :, 1], frames[:, :, 2]
            R, G, B = R >> 5, G >> 5, B >> 5
            return (R << 6) + (G << 3) + B

        batch_size, time_window, height, width, no_channels = frames.shape
        assert no_channels == 3
        # 展平每帧像素以便统计
        frames_flatten = frames.view(batch_size * time_window, height * width, 3)

        # 计算每个像素对应的 bin index（0..511）
        binned_values = get_bin(frames_flatten)
        # 为了对所有帧一次性做 scatter_sum，给每帧分配不同的前缀（frame_bin_prefix）
        frame_bin_prefix = (torch.arange(0, batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned_values = (binned_values + frame_bin_prefix).view(-1)

        # 初始化全局计数向量并把每个像素的 bin 计入（scatter_add）
        histograms = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        histograms.scatter_add_(0, binned_values, torch.ones(len(binned_values), dtype=torch.int32, device=frames.device))

        # 重塑为 [B, T, 512] 并归一化（L2）
        histograms = histograms.view(batch_size, time_window, 512).float()
        histograms_normalized = functional.normalize(histograms, p=2, dim=2)
        return histograms_normalized

    def forward(self, inputs):
        # 先计算每帧的 512-d 直方图向量（L2 归一）
        x = self.compute_color_histograms(inputs)

        batch_size, time_window = x.shape[0], x.shape[1]
        # 计算帧间相似度矩阵（内积）
        similarities = torch.bmm(x, x.transpose(1, 2))  # [batch_size, time_window, time_window]
        # pad 以便提取局部 window
        similarities_padded = functional.pad(similarities, [(self.lookup_window - 1) // 2, (self.lookup_window - 1) // 2])

        # 构建索引并提取局部相似度 [B, T, lookup_window]
        batch_indices = torch.arange(0, batch_size, device=x.device).view([batch_size, 1, 1]).repeat(
            [1, time_window, self.lookup_window])
        time_indices = torch.arange(0, time_window, device=x.device).view([1, time_window, 1]).repeat(
            [batch_size, 1, self.lookup_window])
        lookup_indices = torch.arange(0, self.lookup_window, device=x.device).view([1, 1, self.lookup_window]).repeat(
            [batch_size, time_window, 1]) + time_indices

        similarities = similarities_padded[batch_indices, time_indices, lookup_indices]

        # 可选映射到 output_dim 并 ReLU 激活
        if self.fc is not None:
            return functional.relu(self.fc(similarities))
        return similarities
