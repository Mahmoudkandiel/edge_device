# شرح شامل لـ TPU-MLIR وكيفية نشر موديل جديد على جهاز Sophgo Edge

> **ملاحظة SSH**: للاتصال بالجهاز Edge: `ssh -L 8080:10.0.3.178:8080 -p 26080 linaro@34.47.247.221`

## 🎯 **الكشف التلقائي للـ Backend (ميزة جديدة)**

الكود الآن **يكتشف تلقائياً** Sophon BModel ويستخدمه:
- ✅ **إذا وُجد BModel + Sophon SDK** → يستخدم Sophon TPU (أسرع)
- ⚠️ **إذا لم يوجد BModel لكن PyTorch متوفر** → يستخدم PyTorch/YOLO
- ❌ **إذا لم يوجد أي منهما** → يظهر رسالة خطأ مع تعليمات

**لا حاجة لتغيير الكود!** كل شيء تلقائي.

**التحقق من الإعداد:**
```bash
python3 check_sophon.py
```

هذا السكريبت يتحقق من:
- تثبيت Sophon SDK
- وجود ملف BModel
- هيكل المجلدات
- يعطي توصيات للإعداد

---

## ⚡ **أوامر جاهزة لتحويل YOLOv8 إلى BModel (متوافقة مع الكود - نسخ ولصق مباشرة)**

### **الطريقة السريعة - أوامر جاهزة (تعمل مباشرة مع الكود):**

```bash
# ============================================
# 1. تصدير YOLOv8 إلى ONNX (على جهازك)
# ============================================

pip install --upgrade pip setuptools wheel
pip install "onnx>=1.16.0" "onnxruntime>=1.18.0" "onnxsim>=0.4.36"



python3 << EOF
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, simplify=True)
print("✅ Exported to yolov8n.onnx")
EOF

# ============================================
# 2. تشغيل Docker Container
# ============================================
docker run --privileged --name sophgo_work -v $PWD:/workspace -it sophgo/tpuc_dev:v3.4

# ============================================
# 3. تثبيت TPU-MLIR (داخل Docker)
# ============================================
pip install tpu_mlir[all]

# ============================================
# 4. تحويل ONNX إلى MLIR (داخل Docker)
# ============================================
model_transform \
    --model_name yolov8n \
    --model_def yolov8n.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --pixel_format rgb \
    --mlir yolov8n.mlir

# ============================================
# 5. تحويل MLIR إلى BModel (FP16 - متوافق مع الكود)
# ============================================
model_deploy \
    --mlir yolov8n.mlir \
    --quantize F16 \
    --processor bm1684x \
    --model yolov8n_bm1684x.bmodel

# ============================================
# 6. أو تحويل إلى INT8 (أسرع - يحتاج calibration)
# ============================================
# أ) إنشاء Calibration Table
mkdir dataset
# ضع 100-1000 صورة في dataset/
run_calibration yolov8n.mlir \
    --dataset ./dataset \
    --input_num 100 \
    -o yolov8n_cali_table

# ب) تحويل إلى INT8 (متوافق مع الكود)
model_deploy \
    --mlir yolov8n.mlir \
    --quantize INT8 \
    --calibration_table yolov8n_cali_table \
    --processor bm1684x \
    --model yolov8n_bm1684x.bmodel

# ============================================
# 7. نسخ BModel للجهاز Edge (اسم الملف متوافق مع الكود)
# ============================================
exit  # اخرج من Docker

scp yolov8n_bm1684x.bmodel linaro@34.47.247.221:~/cutrack/models/yolov8n_bm1684x.bmodel
```

### **ملاحظات مهمة:**
- ✅ **اسم الملف**: `yolov8n_bm1684x.bmodel` - هذا هو الاسم الذي يبحث عنه الكود تلقائياً
- ✅ **المسار**: `models/yolov8n_bm1684x.bmodel` - الكود يبحث في مجلد `models/` تلقائياً
- ✅ **لا حاجة لتعديل الكود**: الكود سيكتشف الملف تلقائياً عند تشغيل `python3 main.py`
- ⚠️ **ملفات Git**: ملفات `.bmodel`, `.mlir`, `.onnx`, `.npz` **مستثناة من Git** (في `.gitignore`) لأنها كبيرة جداً
  - ضع ملفات BModel في `models/` محلياً على كل جهاز
  - لا تحاول رفعها إلى Git repository
- **FP16 vs INT8**: 
  - FP16: دقة أعلى، أبطأ قليلاً
  - INT8: أسرع، دقة أقل قليلاً (يحتاج calibration)
- **المعالج**: غير `bm1684x` إلى `bm1688` أو `cv186x` حسب جهازك (لكن غير اسم الملف أيضاً)

---

## ما هو TPU-MLIR؟

TPU-MLIR هو أداة من Sophgo لتحويل نماذج الذكاء الاصطناعي المدربة إلى صيغة **bmodel** التي تعمل بكفاءة على معالجات Tensor Computing Processor (TPU) الخاصة بـ Sophgo.

---

## الخطوات الكاملة لنشر موديل جديد على جهاز Edge

### 🔧 **1. إعداد بيئة العمل (على جهازك المحلي/الخادم)**

#### أ) تثبيت Docker
```bash
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

#### ب) تحميل Docker Image
```bash
docker pull sophgo/tpuc_dev:v3.4
```

#### ج) إنشاء Container
```bash
docker run --privileged --name sophgo_work \
  -v $PWD:/workspace \
  -it sophgo/tpuc_dev:v3.4
```

#### د) تثبيت TPU-MLIR داخل Docker
```bash
pip install tpu_mlir[all]
# أو إذا كان لديك ملف whl:
pip install tpu_mlir-*-py3-none-any.whl[all]
```

---

### 📦 **2. تحضير الموديل الخاص بك**

#### إذا كان الموديل PyTorch (.pt):
```bash
mkdir my_model && cd my_model
# ضع ملف الموديل هنا
cp /path/to/your/model.pt .
mkdir workspace && cd workspace
```

#### إذا كان ONNX (.onnx):
```bash
# استخدمه مباشرة
```

#### إذا كان TensorFlow/Caffe:
```bash
# حوّله لـ ONNX أولاً (راجع Appendix.01 في الوثائق)
```

---

### 🔄 **3. تحويل الموديل إلى MLIR**

#### مثال PyTorch:
```bash
model_transform \
    --model_name my_model \
    --model_def ../model.pt \
    --input_shapes [[1,3,224,224]] \
    --mean 123.675,116.28,103.53 \
    --scale 0.0171,0.0175,0.0174 \
    --pixel_format rgb \
    --test_input /path/to/test_image.jpg \
    --test_result my_model_outputs.npz \
    --mlir my_model.mlir
```

#### مثال ONNX:
```bash
model_transform \
    --model_name my_model \
    --model_def ../model.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --pixel_format rgb \
    --test_input ../test.jpg \
    --test_result my_model_outputs.npz \
    --mlir my_model.mlir
```

**ملاحظات مهمة:**
- `--mean` و `--scale`: قيم preprocessing الخاصة بموديلك
- `--input_shapes`: حجم الـ input المتوقع
- `--pixel_format`: rgb أو bgr حسب موديلك

---

### ⚙️ **4. تحويل MLIR إلى BModel**

#### أ) للحصول على موديل FP16 (دقة عالية):
```bash
model_deploy \
    --mlir my_model.mlir \
    --quantize F16 \
    --processor bm1684x \
    --test_input my_model_in_f32.npz \
    --test_reference my_model_outputs.npz \
    --model my_model_bm1684x_f16.bmodel
```

#### ب) للحصول على موديل INT8 (أسرع):

**الخطوة 1: إنشاء Calibration Table**
```bash
# جهّز مجموعة صور (100-1000 صورة)
mkdir dataset
# ضع الصور في dataset/

run_calibration my_model.mlir \
    --dataset ./dataset \
    --input_num 100 \
    -o my_model_cali_table
```

**الخطوة 2: إنشاء BModel**
```bash
model_deploy \
    --mlir my_model.mlir \
    --quantize INT8 \
    --calibration_table my_model_cali_table \
    --processor bm1684x \
    --test_input my_model_in_f32.npz \
    --test_reference my_model_outputs.npz \
    --tolerance 0.85,0.45 \
    --model my_model_bm1684x_int8.bmodel
```

**اختيار المعالج** (`--processor`):
- `bm1684x` - للأجهزة الحديثة
- `bm1688` - للأجهزة الأحدث
- `cv186x` - لأجهزة CV series
- `bm1684` - للأجهزة القديمة

---

### 📤 **5. نقل BModel إلى جهاز Edge**

#### أ) اخرج من Docker أولاً:
```bash
exit
```

#### ب) انقل الملف عبر SCP:
```bash
# من جهازك المحلي
scp my_model/workspace/my_model_bm1684x_f16.bmodel \
    username@edge_device_ip:/home/username/models/

# مثال:
scp my_model_bm1684x_f16.bmodel sophon@192.168.1.100:/home/sophon/
```

---

### 🚀 **6. تشغيل الموديل على جهاز Edge**

#### أ) اتصل بالجهاز:
```bash
ssh username@edge_device_ip
# مثال:
ssh sophon@192.168.1.100
```

#### ب) تثبيت Runtime Environment (مرة واحدة فقط):
```bash
# حمّل libsophon SDK من Sophgo
# ثبّته حسب دليل التثبيت
```

#### ج) اختبار الموديل:

**للتصنيف (Classification):**
```bash
# إذا كانت أدوات tpu_mlir متاحة:
classify_model \
    --model_def my_model_bm1684x_f16.bmodel \
    --input test_image.jpg \
    --output result.jpg
```

**استخدام Python مخصص:**
```python
import sophon.sail as sail
import numpy as np
from PIL import Image

# تحميل الموديل
engine = sail.Engine("my_model_bm1684x_f16.bmodel", 0, sail.IOMode.SYSIO)
graph_name = engine.get_graph_names()[0]
input_name = engine.get_input_names(graph_name)[0]
output_name = engine.get_output_names(graph_name)[0]

# تحضير الصورة
img = Image.open("test.jpg").resize((224, 224))
img_array = np.array(img).astype(np.float32)
img_array = (img_array - [123.675, 116.28, 103.53]) * [0.0171, 0.0175, 0.0174]
img_array = np.transpose(img_array, (2, 0, 1))
img_array = np.expand_dims(img_array, 0)

# الاستدلال
input_data = {input_name: img_array}
output = engine.process(graph_name, input_data)
result = output[output_name]

print("Result:", result)
```

---

## 🎯 **نصائح مهمة**

### لتحسين الأداء:
1. **استخدم INT8** للسرعة إذا كانت الدقة مقبولة
2. **استخدم `--fuse_preprocess`** لدمج المعالجة المسبقة:
```bash
model_deploy \
    --mlir my_model.mlir \
    --quantize F16 \
    --processor bm1684x \
    --fuse_preprocess \
    --model my_model.bmodel
```

### لحل مشاكل الدقة:
```bash
# استخدم Mixed Precision
run_calibration my_model.mlir \
    --dataset ./dataset \
    --input_num 100 \
    --search search_qtable \
    --quantize_table my_qtable \
    -o my_cali_table

model_deploy \
    --quantize_table my_qtable \
    # ... باقي الخيارات
```

### للتحقق من الموديل:
```bash
# على جهاز Edge
model_tool --info my_model.bmodel
```

---

## 📊 **مثال كامل: YOLOv8 (أوامر جاهزة)**

```bash
# ============================================
# الخطوة 1: تصدير YOLOv8 إلى ONNX
# ============================================
# على جهازك (خارج Docker)
python3 << EOF
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, simplify=True)
print("✅ Exported: yolov8n.onnx")
EOF

# ============================================
# الخطوة 2: تشغيل Docker
# ============================================
docker run --privileged --name sophgo_work -v $PWD:/workspace -it sophgo/tpuc_dev:v3.4
pip install tpu_mlir[all]

# ============================================
# الخطوة 3: تحويل ONNX → MLIR
# ============================================
model_transform \
    --model_name yolov8n \
    --model_def yolov8n.onnx \
    --input_shapes [[1,3,640,640]] \
    --mean 0.0,0.0,0.0 \
    --scale 0.0039216,0.0039216,0.0039216 \
    --pixel_format rgb \
    --mlir yolov8n.mlir

# ============================================
# الخطوة 4: تحويل MLIR → BModel (FP16 - متوافق مع الكود)
# ============================================
model_deploy \
    --mlir yolov8n.mlir \
    --quantize F16 \
    --processor bm1684x \
    --model yolov8n_bm1684x.bmodel

# ============================================
# الخطوة 5: أو INT8 (أسرع - يحتاج calibration)
# ============================================
# أ) Calibration
mkdir dataset
# ضع صور في dataset/ (100-1000 صورة)
run_calibration yolov8n.mlir \
    --dataset ./dataset \
    --input_num 100 \
    -o yolov8n_cali_table

# ب) Deploy INT8 (متوافق مع الكود - نفس الاسم)
model_deploy \
    --mlir yolov8n.mlir \
    --quantize INT8 \
    --calibration_table yolov8n_cali_table \
    --processor bm1684x \
    --model yolov8n_bm1684x.bmodel

# ============================================
# الخطوة 6: نقل للجهاز Edge (اسم متوافق مع الكود)
# ============================================
exit  # اخرج من Docker
scp yolov8n_bm1684x.bmodel linaro@34.47.247.221:~/cutrack/models/yolov8n_bm1684x.bmodel
```

---

## 🔍 **استكشاف الأخطاء**

| المشكلة | الحل |
|---------|------|
| خطأ في التحويل | تحقق من شكل الـ input وقيم preprocessing |
| دقة منخفضة في INT8 | زد عدد صور calibration أو استخدم mixed precision |
| بطء في التشغيل | استخدم INT8 أو فعّل `--fuse_preprocess` |
| خطأ في الذاكرة | قلل batch size أو استخدم `--dynamic` |

---

## 🚀 **الأوامر التي ستنفذ على جهاز Edge فقط**

> **ملاحظة**: هذه الأوامر تنفذ **فقط على جهاز Edge** بعد تحويل الموديل ونقله.

---

### **الخطوة 1: إعداد Sophon SDK (مرة واحدة فقط)**

```bash
# تثبيت Sophon SDK (حمّله من https://developer.sophgo.com/)
cd /path/to/sophon_sdk
sudo ./install.sh

# أو إعداد يدوي:
export LD_LIBRARY_PATH=/opt/sophon/libsophon/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/sophon/sophon-sail/python3/lib:$PYTHONPATH

# إضافة للملف لتكون دائمة:
echo 'export LD_LIBRARY_PATH=/opt/sophon/libsophon/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export PYTHONPATH=/opt/sophon/sophon-sail/python3/lib:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

# التحقق من التثبيت:
python3 -c "import sophon.sail as sail; print('✅ Sophon SDK OK')"
```

---

### **الخطوة 2: إعداد CuTrack**

```bash
# الانتقال لمجلد المشروع
cd ~/cutrack

# إنشاء virtual environment
python3 -m venv .venv
source .venv/bin/activate

# تثبيت التبعيات
pip install -r requirements_sophon.txt

# إنشاء مجلد النماذج (إذا لم يكن موجود)
mkdir -p models
```

---

### **الخطوة 3: التحقق من BModel**

```bash
# التحقق من وجود الملف
ls -lh models/yolov8n_bm1684x.bmodel

# يجب أن ترى الملف موجود
# إذا لم يكن موجود، انقله من جهازك المحلي:
# scp yolov8n_bm1684x.bmodel linaro@34.47.247.221:~/cutrack/models/
```

---

### **الخطوة 4: التحقق من الإعداد**

```bash
# تشغيل سكريبت التحقق (جديد في v1.1.0)
python3 check_sophon.py
```

**النتيجة المتوقعة:**
```
============================================================
CuTrack - Sophon Deployment Check
============================================================

✅ Sophon SDK installed successfully!
✅ BModel found: models/yolov8n_bm1684x.bmodel
✅ Models directory exists
✅ Output directories exist
✅ Everything looks good! You're ready to deploy.
```

**ملاحظة**: هذا السكريبت متوفر في الإصدار v1.1.0+ ويوفر فحص شامل للإعداد.

---

### **الخطوة 5: تشغيل CuTrack**

```bash
# تفعيل virtual environment (إذا لم يكن مفعّل)
source .venv/bin/activate

# تشغيل CuTrack
python3 main.py
```

**ستظهر رسالة (الكشف التلقائي):**
```
🚀 Enhanced Customer Foot Traffic Analyzer Initialized!
✅ Sophon TPU detected - Using BModel: models/yolov8n_bm1684x.bmodel
```

**ملاحظة**: إذا لم يكتشف Sophon، سيظهر:
```
ℹ️  Sophon SDK not available - using PyTorch/YOLO (if available)
```

الكود سيعمل تلقائياً مع أي backend متوفر.

**استخدام CuTrack:**
1. اختر `1` لتكوين ROI (مناطق الاهتمام)
2. اختر `2` لتحليل فيديو مسجل
3. اختر `3` لتحليل مباشر من الكاميرا

---

### **الخطوة 6: تشغيل كخدمة (اختياري)**

```bash
# إنشاء ملف الخدمة
sudo nano /etc/systemd/system/cutrack.service
```

**أضف المحتوى التالي (عدّل المسارات حسب اسم المستخدم):**
```ini
[Unit]
Description=CuTrack Customer Traffic Analyzer
After=network.target

[Service]
Type=simple
User=linaro
WorkingDirectory=/home/linaro/cutrack
Environment="PATH=/home/linaro/cutrack/.venv/bin"
Environment="LD_LIBRARY_PATH=/opt/sophon/libsophon/lib:$LD_LIBRARY_PATH"
Environment="PYTHONPATH=/opt/sophon/sophon-sail/python3/lib:$PYTHONPATH"
ExecStart=/home/linaro/cutrack/.venv/bin/python3 main.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**تفعيل وتشغيل الخدمة:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable cutrack.service
sudo systemctl start cutrack.service

# التحقق من الحالة
sudo systemctl status cutrack.service

# عرض السجلات
sudo journalctl -u cutrack.service -f
```

---

## 📋 **ملخص الأوامر السريعة (على جهاز Edge فقط)**

```bash
# 1. إعداد Sophon SDK (مرة واحدة)
export LD_LIBRARY_PATH=/opt/sophon/libsophon/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/sophon/sophon-sail/python3/lib:$PYTHONPATH
python3 -c "import sophon.sail as sail; print('OK')"

# 2. إعداد CuTrack
cd ~/cutrack
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_sophon.txt

# 3. التحقق من BModel
ls -lh models/yolov8n_bm1684x.bmodel

# 4. التحقق من الإعداد
python3 check_sophon.py

# 5. تشغيل CuTrack
source .venv/bin/activate
python3 main.py
```

---

## ⚙️ **كيف يعمل الكشف التلقائي (v1.1.0+)**

CuTrack **يُكتشف تلقائياً** Sophon BModel ويستخدمه:

### **آلية الكشف:**

```
PersonDetector.__init__()
    ↓
فحص Sophon BModel
    ├─ موجود + Sophon SDK متوفر → استخدام PersonDetectorSOPHON (TPU)
    └─ غير موجود → فحص PyTorch
        ├─ متوفر → استخدام YOLO (GPU/CPU)
        └─ غير متوفر → رسالة خطأ مع تعليمات
```

### **أماكن البحث عن BModel:**

الكود يبحث تلقائياً في:
1. المسار المحدد في `Config.SOPHON.BMODEL_PATH` (إن وُجد)
2. `models/yolov8n_bm1684x.bmodel`
3. `models/yolov8n.bmodel`
4. `models/yolov8_bm1684x.bmodel`
5. `models/yolov8.bmodel`

### **الملفات الجديدة:**

- `modules/demographics_sophon.py` - كاشف Sophon TPU
- `utils/sophon_utils.py` - أدوات Sophon SDK
- `check_sophon.py` - سكريبت التحقق من الإعداد
- `requirements_sophon.txt` - تبعيات محسّنة لـ Sophon

**لا حاجة لتغيير الكود!** كل شيء تلقائي ومتوافق مع الإصدارات السابقة.

---

## 🎯 **نصائح للأداء الأمثل**

### 1. استخدم INT8 بدلاً من FP16:
```bash
# عند التحويل، استخدم INT8 للحصول على أسرع أداء
model_deploy --quantize INT8 --calibration_table cali_table ...
```

### 2. اضبط إعدادات المعالجة:
```python
# في config/settings.py
Config.VIDEO.FRAME_SKIP = 5  # معالجة كل إطار خامس (أسرع)
Config.ANALYSIS.SAVE_VIDEO_OUTPUT = False  # توفير الذاكرة
```

### 3. راقب الأداء:
```bash
# على جهاز Edge
htop  # لمراقبة استخدام CPU/ذاكرة
# أو استخدم أدوات Sophon المخصصة
```

---

## 📞 **الدعم والمساعدة**

- **مشاكل في التحويل**: راجع قسم "استكشاف الأخطاء" أعلاه
- **مشاكل في النشر**: راجع `SOPHON_CHANGES.md` للتفاصيل التقنية
- **تحقق من الإعداد**: شغّل `python3 check_sophon.py` (v1.1.0+)
- **مثال كامل**: راجع قسم "مثال كامل: YOLOv8" أعلاه
- **الكود الجديد**: راجع `modules/demographics.py` لآلية الكشف التلقائي

---



