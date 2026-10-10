# YOLOv8 后台训练

分类器已经训完。这一次只在 Kaggle 后台训练检测器，拿到 `best.pt`。本地已有的 `weights/best_model.pth` 不要换。

整次运行最长 12 小时。Cell 5 本身大约 8 到 12 小时。页面可以关掉，训练在后台继续。

## 1. 要导入的文件

只导入这一个笔记本：

| 本机文件 | 在 Kaggle 里怎么用 |
| --- | --- |
| `kaggle/tt100k_yolo_train.ipynb` | **File → Import Notebook** |

不要导入这些：

- `kaggle/tt100k_two_stage.ipynb`（那是含分类训练的完整版）
- `weights/best_model.pth`
- `weights/dataset.yaml`
- `demo.py`

数据集不是上传文件。在新笔记本右侧 **Add Input → Datasets**，搜索并添加 [braunge/tt100k](https://www.kaggle.com/datasets/braunge/tt100k)。挂上之后，后台运行也能读到 `/kaggle/input/datasets/braunge/tt100k`。

## 2. 新建并保存版本

1. 打开 [kaggle.com/code](https://www.kaggle.com/code)，**File → Import Notebook**，选择 `kaggle/tt100k_yolo_train.ipynb`。
2. **Add Input** 添加上面的数据集。
3. 右侧 **Notebook options**：
   - Accelerator：**GPU T4 x2**
   - Internet：**On**（Cell 1 要装 ultralytics，Cell 5 要下载 `yolov8s.pt`）
4. 若上一个交互会话还开着，先 **Stop Session**。否则两份 GPU 同时占额度。停掉之后，那个会话里的 `/kaggle/working/tt100k45` 会消失；新笔记本的 Cell 4 会重新建一份。
5. 不要在编辑器里先把包装好再指望后台沿用。**Save Version → Save & Run All** 会另开一台空机器，从 Cell 1 重新执行。
6. 保存版本时再确认一次：加速器是 **GPU T4 x2**，Internet 是 **On**。

不需要在编辑器里逐格运行。若想先看类别名对不对，可以只跑到 Cell 3，确认日志里有这两行后再停掉交互会话，然后对同一份笔记本做 Save & Run All：

```text
类别名来自 .../YOLOv8_TT100K.yaml，共 50 类
类别核对通过: pl80 pn pl50 il110
```

对不上时笔记本会自己停止，不会进入 30 轮训练。

## 3. 六个格子在做什么

| 单元 | 在做什么 | 正常时你会看到 |
| --- | --- | --- |
| Cell 1 | 安装 ultralytics 等 | 「依赖已安装」 |
| Cell 2 | 检查 GPU，打印数据目录 | `gpu_count 2`，以及 `mydata/images` |
| Cell 3 | 读 50 个类别名并画柱状图 | `类别核对通过: pl80 pn pl50 il110` |
| Cell 4 | 建检测数据，图片用符号链接 | `图片链接方式: symlink`，以及「检测数据核对通过」 |
| Cell 5 | 训练 YOLOv8s，30 轮，双卡 | 训练日志。结束后把 `best.pt` 复制到 `/kaggle/working/best.pt` |
| Cell 6 | 打印要下载的路径 | `best.pt` 和 `dataset.yaml` 都是 `[就绪]` |

Cell 4 如果不能建符号链接，会直接报错并停止，避免把两万张原图复制进工作目录。

## 4. 下载

打开**这一版**的 Output，不是编辑器里旧会话的 Output。文件在 `/kaggle/working/` 根目录。

| 下载这个 | 放到本地 |
| --- | --- |
| `best.pt` | `weights/best.pt` |
| `dataset.yaml` | `weights/dataset.yaml`（打开确认第一项是 `pl80`，一共 50 项，再覆盖） |

`weights/best_model.pth` 保持上次下载的文件，不要覆盖。

12 小时到点、Cell 6 没跑完时，根目录可能还没有 `best.pt`。改从 Output 下载：

```text
yolo_runs/tt100k45/weights/best.pt
```

放到本地时仍命名为 `weights/best.pt`。

不要下载 `tt100k45/`，也不要下载整个 `yolo_runs/`。

下完后本地应是：

```text
Project1/
  demo.py
  weights/
    best.pt
    best_model.pth
    dataset.yaml
```

在项目根目录执行：

```bash
pip install -r requirements.txt
python demo.py --yolo weights/best.pt --resnet weights/best_model.pth --data weights/dataset.yaml
```

浏览器打开 http://127.0.0.1:8000 。服务只监听本机。
