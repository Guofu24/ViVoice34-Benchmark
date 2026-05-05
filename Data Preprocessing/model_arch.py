import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as kaldi
import yaml
import os

class ASP(nn.Module):
    def __init__(self, in_planes, acoustic_dim):
        super(ASP, self).__init__()
        outmap_size = int(acoustic_dim / 8)
        self.out_dim = in_planes * 8 * outmap_size * 2
        self.attention = nn.Sequential(
            nn.Conv1d(in_planes * 8 * outmap_size, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, in_planes * 8 * outmap_size, kernel_size=1),
            nn.Softmax(dim=2),
        )

    def forward(self, x):
        x = x.reshape(x.size()[0], -1, x.size()[-1])
        w = self.attention(x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-5))
        x = torch.cat((mu, sg), 1)
        x = x.view(x.size()[0], -1)
        return x


class SimAMBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ConvLayer, NormLayer, in_planes, planes, stride=1, block_id=1):
        super(SimAMBasicBlock, self).__init__()
        self.conv1 = ConvLayer(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.conv2 = ConvLayer(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = NormLayer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                ConvLayer(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                NormLayer(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.SimAM(out)
        out += self.downsample(x)
        out = self.relu(out)
        return out

    def SimAM(self, X, lambda_p=1e-4):
        n = X.shape[2] * X.shape[3] - 1
        d = (X - X.mean(dim=[2, 3], keepdim=True)).pow(2)
        v = d.sum(dim=[2, 3], keepdim=True) / n
        E_inv = d / (4 * (v + lambda_p)) + 0.5
        return X * self.sigmoid(E_inv)


class ResNet(nn.Module):
    def __init__(self, in_planes, block, num_blocks, in_ch=1, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.NormLayer = nn.BatchNorm2d
        self.ConvLayer = nn.Conv2d
        self.conv1 = self.ConvLayer(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.NormLayer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, in_planes * 8, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.ConvLayer, self.NormLayer, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def SimAM_ResNet34(in_planes): return ResNet(in_planes, SimAMBasicBlock, [3, 4, 6, 3])
def SimAM_ResNet100(in_planes): return ResNet(in_planes, SimAMBasicBlock, [6, 16, 24, 3])


class SimAM_ResNet_ASP(nn.Module):
    def __init__(self, depth=34, in_planes=64, embed_dim=256, acoustic_dim=80, dropout=0):
        super(SimAM_ResNet_ASP, self).__init__()
        if depth == 100:
            self.front = SimAM_ResNet100(in_planes)
        else:
            self.front = SimAM_ResNet34(in_planes)
        self.pooling = ASP(in_planes, acoustic_dim)
        self.bottleneck = nn.Linear(self.pooling.out_dim, embed_dim)
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.front(x.unsqueeze(dim=1))
        x = self.pooling(x)
        if self.drop:
            x = self.drop(x)
        x = self.bottleneck(x)
        return x


# PHẦN 2: HÀM LOAD MODEL & XỬ LÝ AUDIO
def load_model(config_path, checkpoint_path, device='cpu'):
    print(f"Loading model from: {checkpoint_path}")

    # 1. Load Config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Xác định kiến trúc (ResNet34 hay 100)
    depth = 34  # Mặc định
    if 'resnet100' in str(config_path) or 'resnet100' in str(checkpoint_path):
        depth = 100
    if config.get('num_blocks') == [6, 16, 24, 3]:
        depth = 100

    print(f"-> Khởi tạo SimAM_ResNet{depth}_ASP...")

    model = SimAM_ResNet_ASP(
        depth=depth,
        in_planes=config.get("in_planes", 64),
        embed_dim=config.get("embed_dim", 256),
        acoustic_dim=config.get("acoustic_dim", 80),
        dropout=config.get("dropout", 0.0)
    )

    # 3. Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Lấy state_dict từ các key phổ biến
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 4. XỬ LÝ MAPPING KEY
    new_state = {}

    # Danh sách các key đặc trưng của ResNet cần thêm prefix 'front.'
    resnet_prefixes = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")

    for k, v in state_dict.items():
        # Bước 1: Làm sạch prefix rác từ Lightning/Train
        # Xóa 'backbone.', 'model.', 'module.' nếu có
        clean_k = k
        for prefix in ["backbone.", "model.", "module."]:
            if clean_k.startswith(prefix):
                clean_k = clean_k[len(prefix):]  # Cắt bỏ prefix

        # Bước 2: Map vào đúng vị trí trong Model hiện tại
        # Nếu là layer của ResNet (vd: layer1.0.conv1...) -> Thêm 'front.'
        if clean_k.startswith(resnet_prefixes):
            new_key = f"front.{clean_k}"
        else:
            # Các layer khác (pooling, bottleneck, loss...) giữ nguyên
            new_key = clean_k

        new_state[new_key] = v

    # 5. Load vào Model
    msg = model.load_state_dict(new_state, strict=False)
    missing_critical = [k for k in msg.missing_keys if "num_batches_tracked" not in k]

    if len(missing_critical) > 0:
        print(f"⚠️ CẢNH BÁO: Vẫn thiếu {len(missing_critical)} keys: {missing_critical[:3]}...")
    else:
        print("✅ Load weights thành công (Logic: Auto-Prefix 'front.')")

    model.to(device)
    model.eval()

    return model


def compute_fbank(waveform, sample_rate=16000, num_mel_bins=80):
    waveform = waveform * (1 << 15)
    # Thêm dither=0.0 để loại bỏ ngẫu nhiên, vì mỗi lần chạy dither=1.0 xem thêm nhiễu random
    feat = kaldi.fbank(waveform, num_mel_bins=num_mel_bins,
                       frame_length=25, frame_shift=10,
                       sample_frequency=sample_rate, window_type='hamming',
                       dither=0.0)
    feat = feat - torch.mean(feat, 0)
    return feat.unsqueeze(0)