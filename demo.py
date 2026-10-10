"""本地两阶段路标推理：YOLOv8 检测框，手写 ResNet-18 识别含义。

把 Kaggle 笔记本导出的三个文件放在 weights/ 后启动：

    pip install -r requirements.txt
    python demo.py --yolo weights/best.pt --resnet weights/best_model.pth --data weights/dataset.yaml

浏览器打开 http://127.0.0.1:8000 ，上传街景图。服务只监听本机，不挂到局域网。
预处理 Resize(224) + ToTensor + ImageNet Normalize 只作用于裁剪出的路标小图。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DET_CONF = 0.25
MIN_BOX = 16
YOLO_PROBE_CONF = 0.001

# 与训练笔记本写入 dataset.yaml 的规则一致。推理优先使用 yaml / checkpoint 里的含义。
MEANING_TABLE = {
    "i1": "步行",
    "i2": "非机动车行驶",
    "i3": "环岛行驶",
    "i4": "机动车行驶",
    "i5": "靠右侧道路行驶",
    "i6": "靠左侧道路行驶",
    "i7": "立体交叉直行和右转弯行驶",
    "i8": "立体交叉直行和左转弯行驶",
    "i9": "鸣喇叭",
    "i10": "向右转弯",
    "i11": "向左向右转弯",
    "i12": "向左转弯",
    "i13": "直行",
    "i14": "直行和向右转弯",
    "i15": "直行和向左转弯",
    "il50": "最低限速50",
    "il60": "最低限速60",
    "il70": "最低限速70",
    "il80": "最低限速80",
    "il90": "最低限速90",
    "il100": "最低限速100",
    "il110": "最低限速110",
    "io": "其他指示标志",
    "ip": "人行横道",
    "p1": "禁止超车",
    "p2": "禁止畜力车进入",
    "p3": "禁止大型客车驶入",
    "p4": "禁止电动三轮车驶入",
    "p5": "禁止掉头",
    "p6": "禁止非机动车进入",
    "p7": "禁止载货汽车左转",
    "p8": "禁止汽车拖、挂车驶入",
    "p9": "禁止行人进入",
    "p10": "禁止机动车驶入",
    "p11": "禁止鸣喇叭",
    "p12": "禁止二轮摩托车驶入",
    "p13": "禁止某两种车驶入",
    "p14": "禁止直行",
    "p15": "禁止人力车进入",
    "p16": "禁止人力货运三轮车进入",
    "p17": "禁止人力客运三轮车进入",
    "p18": "禁止三轮车机动车通行",
    "p19": "禁止向右转弯",
    "p20": "禁止向左向右转弯",
    "p21": "禁止直行和向右转弯",
    "p23": "禁止向左转弯",
    "p26": "禁止载货汽车驶入",
    "p27": "禁止运输危险物品车辆驶入",
    "p28": "禁止直行和向左转弯",
    "pb": "禁止通行",
    "pbp": "禁止自行车和行人通行",
    "pg": "减速让行",
    "ph": "限高",
    "pn": "禁止停车",
    "pne": "禁止驶入",
    "po": "其他禁止标志",
    "pa": "限制轴重",
    "pm": "限重",
    "pr": "解除限速",
    "pr40": "解除限速40",
    "w": "警告标志",
    "ps": "停车让行",
    "w13": "十字交叉",
    "w32": "施工",
    "w55": "注意儿童",
    "w57": "注意行人",
    "w59": "注意合流",
    "wo": "其他警告标志",
}

CROP_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>交通路标识别</title>
  <style>
    body { font-family: "Microsoft YaHei", sans-serif; margin: 32px auto; max-width: 980px; color: #1c1c1c; }
    h1 { font-size: 24px; margin-bottom: 8px; }
    p { color: #444; }
    form { margin: 16px 0 24px; }
    img { max-width: 100%; height: auto; border: 1px solid #ddd; }
    table { border-collapse: collapse; width: 100%; margin-top: 16px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background: #f4f4f4; }
    .err { color: #a12626; }
  </style>
</head>
<body>
  <h1>交通路标识别</h1>
  <p>上传一张街景图。服务先检测路标位置，再识别每个路标的含义，并在原图上画框。</p>
  <form id="form">
    <input type="file" name="file" accept="image/*" required>
    <button type="submit">识别</button>
  </form>
  <div id="message"></div>
  <img id="preview" alt="" hidden>
  <table id="table" hidden>
    <thead>
      <tr><th>含义</th><th>类别代码</th><th>位置</th><th>检测置信度</th><th>分类置信度</th></tr>
    </thead>
    <tbody></tbody>
  </table>
  <script>
    function esc(text) {
      return String(text).replace(/[&<>"]/g, (ch) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"
      }[ch]));
    }
    document.getElementById("form").addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = document.getElementById("message");
      const preview = document.getElementById("preview");
      const table = document.getElementById("table");
      message.className = "";
      message.textContent = "正在识别...";
      preview.hidden = true;
      table.hidden = true;
      try {
        const response = await fetch("/predict", { method: "POST", body: new FormData(event.target) });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || "识别失败");
        }
        preview.src = "data:image/jpeg;base64," + data.image_base64;
        preview.hidden = false;
        const body = table.querySelector("tbody");
        body.innerHTML = data.signs.map((sign) => `<tr>
          <td>${esc(sign.meaning)}</td>
          <td>${esc(sign.code)}</td>
          <td>${sign.bbox.map((v) => Math.round(v)).join(", ")}</td>
          <td>${Number(sign.det_conf).toFixed(3)}</td>
          <td>${Number(sign.cls_conf).toFixed(3)}</td>
        </tr>`).join("");
        table.hidden = false;
        message.textContent = data.signs.length ? `检测到 ${data.signs.length} 个路标` : "未检测到路标";
      } catch (error) {
        message.className = "err";
        message.textContent = error.message;
      }
    });
  </script>
</body>
</html>
"""


class BasicBlock(nn.Module):
    """ResNet 基础残差块。通道或步长变化时，用 1x1 卷积把捷径分支对齐到主分支。"""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride=block_stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


class UnicodeJSONResponse(JSONResponse):
    def render(self, content):
        return json.dumps(content, ensure_ascii=False).encode("utf-8")


def meaning_of(code):
    code = str(code)
    if code in MEANING_TABLE:
        return MEANING_TABLE[code]
    matched = re.fullmatch(r"pl(\d+(?:\.\d+)?)", code)
    if matched:
        return f"限速{matched.group(1)}"
    matched = re.fullmatch(r"il(\d+(?:\.\d+)?)", code)
    if matched:
        return f"最低限速{matched.group(1)}"
    matched = re.fullmatch(r"ph(\d+(?:\.\d+)?)", code)
    if matched:
        return f"限高{matched.group(1)}米"
    matched = re.fullmatch(r"pm(\d+(?:\.\d+)?)", code)
    if matched:
        return f"限重{matched.group(1)}吨"
    matched = re.fullmatch(r"pw(\d+(?:\.\d+)?)", code)
    if matched:
        return f"限宽{matched.group(1)}米"
    matched = re.fullmatch(r"pr(\d+(?:\.\d+)?)", code)
    if matched:
        return f"解除限速{matched.group(1)}"
    matched = re.fullmatch(r"pa(\d+(?:\.\d+)?)", code)
    if matched:
        return f"限制轴重{matched.group(1)}吨"
    return code


def as_string_list(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        ordered = sorted((int(key), str(item)) for key, item in value.items())
        return [item for _, item in ordered]
    raise TypeError(f"无法解析名单: {type(value)}")


def load_dataset_spec(path):
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "names" not in data:
        raise ValueError(f"{path} 缺少 names")
    class_names = as_string_list(data["names"])
    if "meanings" in data:
        meanings = as_string_list(data["meanings"])
        if len(meanings) != len(class_names):
            raise ValueError("dataset.yaml 中 meanings 与 names 长度不一致")
    else:
        meanings = [meaning_of(name) for name in class_names]
    return class_names, meanings


def strip_module_prefix(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        name = key
        while name.startswith("module."):
            name = name[len("module.") :]
        cleaned[name] = value
    return cleaned


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_font(size=18):
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def clip_box(bbox, width, height, min_side):
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(round(x2))))
    y2 = max(0, min(height, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    if min(x2 - x1, y2 - y1) < min_side:
        return None
    return x1, y1, x2, y2


def preprocess_crop(crop):
    """只预处理路标小图，不处理整张街景。"""
    return CROP_TRANSFORM(crop)


class Predictor:
    def __init__(self, yolo_path, resnet_path, data_path, det_conf=DET_CONF, min_box=MIN_BOX):
        self.det_conf = det_conf
        self.min_box = min_box
        self.device = select_device()
        self.font = load_font()
        self.lock = threading.Lock()
        yaml_names, yaml_meanings = load_dataset_spec(data_path)
        checkpoint = torch_load(resnet_path, self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            checkpoint_names = checkpoint.get("class_names")
            checkpoint_meanings = checkpoint.get("meanings")
            normalize = checkpoint.get("normalize")
        else:
            state_dict = checkpoint
            checkpoint_names = None
            checkpoint_meanings = None
            normalize = None
        if normalize is not None:
            mean = tuple(normalize["mean"])
            std = tuple(normalize["std"])
            if mean != IMAGENET_MEAN or std != IMAGENET_STD:
                raise RuntimeError("checkpoint 的 Normalize 与本地预处理不一致")
        if checkpoint_names is not None and [str(name) for name in checkpoint_names] != yaml_names:
            raise RuntimeError(
                "dataset.yaml 的 names 与 best_model.pth 的 class_names 不一致。"
                "请使用同一次训练导出的 best.pt、best_model.pth 和 dataset.yaml。"
            )
        self.class_names = yaml_names
        if checkpoint_meanings is not None and [str(item) for item in checkpoint_meanings] != yaml_meanings:
            print("警告: checkpoint 与 dataset.yaml 的含义不一致，界面使用 dataset.yaml。")
        self.meanings = yaml_meanings
        state_dict = strip_module_prefix(state_dict)
        self.resnet = ResNet18(num_classes=len(self.class_names)).to(self.device)
        self.resnet.load_state_dict(state_dict)
        self.resnet.eval()
        weight = self.resnet.fc.weight
        if weight.shape[0] != len(self.class_names):
            raise RuntimeError("ResNet 输出维度与类别数不一致")
        from ultralytics import YOLO

        self.yolo = YOLO(str(yolo_path))
        print(f"设备: {self.device}")
        print(f"类别数: {len(self.class_names)}")

    def detect_boxes(self, image):
        device_arg = 0 if self.device.type == "cuda" else "cpu"
        results = self.yolo.predict(
            source=image,
            conf=YOLO_PROBE_CONF,
            imgsz=640,
            verbose=False,
            device=device_arg,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        boxes = []
        for box, score in zip(xyxy, scores):
            boxes.append({"bbox": [float(v) for v in box.tolist()], "det_conf": float(score)})
        return boxes

    def predict(self, image):
        # 1. 接收任意尺寸的原始图片。后续几何操作都回到这张图的坐标系。
        if not isinstance(image, Image.Image):
            raise TypeError("image 必须是 PIL.Image")
        image = image.convert("RGB")
        width, height = image.size

        # 2. 把整张原图送入 YOLO，得到边界框和检测置信度。
        detected = self.detect_boxes(image)

        # 3. 过滤低置信度框。
        detected = [box for box in detected if box["det_conf"] >= self.det_conf]

        # 4. 从原图裁剪路标。坐标越界先截断，过小的框直接丢弃。
        crops = []
        for box in detected:
            clipped = clip_box(box["bbox"], width, height, self.min_box)
            if clipped is None:
                continue
            tile = image.crop(clipped)
            crops.append({**box, "bbox": list(clipped), "tile": tile})

        # 5. 每张小图单独预处理：Resize 到 224，再 ToTensor，再 ImageNet Normalize。
        # 6. 预处理后的小图进入 ResNet，得到类别和分类置信度。
        if crops:
            batch = torch.stack([preprocess_crop(item["tile"]) for item in crops], dim=0)
            batch = batch.to(self.device)
            with torch.no_grad():
                logits = self.resnet(batch)
                probabilities = torch.softmax(logits, dim=1)
                confidences, class_ids = probabilities.max(dim=1)
            for item, class_id, confidence in zip(crops, class_ids.tolist(), confidences.tolist()):
                item["class_id"] = int(class_id)
                item["cls_conf"] = float(confidence)
                item["code"] = self.class_names[item["class_id"]]
                item["meaning"] = self.meanings[item["class_id"]]

        # 7. 在原图上画框，标注含义和分类置信度。
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        palette = [(220, 68, 55), (25, 118, 210), (46, 125, 50), (245, 124, 0), (123, 31, 162), (0, 121, 107)]
        signs = []
        for index, item in enumerate(crops):
            color = palette[index % len(palette)]
            x1, y1, x2, y2 = item["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f"{item['meaning']} {item['cls_conf']:.2f}"
            text_box = draw.textbbox((0, 0), text, font=self.font)
            text_w = text_box[2] - text_box[0]
            text_h = text_box[3] - text_box[1]
            text_y = y1 - text_h - 6
            if text_y < 0:
                text_y = min(y1 + 2, max(0, height - text_h - 4))
            draw.rectangle([x1, text_y, min(width, x1 + text_w + 8), text_y + text_h + 4], fill=color)
            draw.text((x1 + 4, text_y + 2), text, fill=(255, 255, 255), font=self.font)
            signs.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "class_id": item["class_id"],
                    "code": item["code"],
                    "meaning": item["meaning"],
                    "det_conf": round(item["det_conf"], 4),
                    "cls_conf": round(item["cls_conf"], 4),
                }
            )

        # 8. 返回画好框的图片，以及每个路标的位置、含义和置信度。
        return annotated, signs


def encode_jpeg(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


app = FastAPI(title="交通路标识别")
STATE = {"predictor": None, "args": None}


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/predict", response_class=UnicodeJSONResponse)
async def predict_api(file: UploadFile = File(...)):
    predictor = STATE["predictor"]
    if predictor is None:
        raise HTTPException(status_code=503, detail="模型尚未加载")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="上传文件为空")
    try:
        image = Image.open(io.BytesIO(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法解析为图片") from exc
    with predictor.lock:
        annotated, signs = predictor.predict(image)
    return {
        "signs": signs,
        "width": annotated.width,
        "height": annotated.height,
        "image_base64": encode_jpeg(annotated),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="本地交通路标识别服务")
    parser.add_argument("--yolo", default="weights/best.pt", help="YOLOv8 best.pt")
    parser.add_argument("--resnet", default="weights/best_model.pth", help="ResNet-18 best_model.pth")
    parser.add_argument("--data", default="weights/dataset.yaml", help="训练导出的 dataset.yaml")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--det-conf", type=float, default=DET_CONF)
    parser.add_argument("--min-box", type=int, default=MIN_BOX)
    return parser.parse_args()


def require_weight(path):
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"找不到 {file_path}。请把 Kaggle Output 中的 best.pt、best_model.pth、dataset.yaml 放到 weights/。"
        )
    return file_path


def main():
    args = parse_args()
    yolo_path = require_weight(args.yolo)
    resnet_path = require_weight(args.resnet)
    data_path = require_weight(args.data)
    STATE["args"] = args
    STATE["predictor"] = Predictor(
        yolo_path,
        resnet_path,
        data_path,
        det_conf=args.det_conf,
        min_box=args.min_box,
    )
    print(f"只在本机提供服务: http://127.0.0.1:{args.port}")
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
