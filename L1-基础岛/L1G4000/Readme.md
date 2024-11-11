![image-20241111201612631](./assets/image-20241111201612631.png)

# 一、LlamaIndex+InternLM API 实践

> **任务要求**：基于 LlamaIndex 构建自己的 RAG 知识库，寻找一个问题 A 在使用 LlamaIndex 之前 浦语 API 不会回答，借助 LlamaIndex 后 浦语 API 具备回答 A 的能力，截图保存。

## 1.创建新的conda环境，命名为 `llamaindex`

```bash
conda create -n llamaindex python=3.10
```

![image-20241111202404937](./assets/image-20241111202404937.png)

## 2.激活 `llamaindex` 

```bash
conda activate llamaindex
```

![image-20241111202850152](./assets/image-20241111202850152.png)

## 3.安装相关基础依赖 **python** 虚拟环境

```bash
pip install einops==0.7.0 protobuf==5.26.1
```

![image-20241111203009267](./assets/image-20241111203009267.png)

## 4.安装 Llamaindex

```bash
pip install llama-index==0.11.20
pip install llama-index-llms-replicate==0.3.0
pip install llama-index-llms-openai-like==0.2.0
pip install llama-index-embeddings-huggingface==0.3.1
pip install llama-index-embeddings-instructor==0.2.1
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```

> 此处没图……ps.这些个包安装起来好慢……

## 5.下载词向量模型`Sentence Transformer`

```bash
cd ~
mkdir llamaindex_demo
mkdir model
cd ~/llamaindex_demo
touch download_hf.py
```

编辑`download_hf.py`代码

```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/model/sentence-transformer')
```

![image-20241111212458090](./assets/image-20241111212458090.png)

运行`download_hf.py`下载模型

```python
python download_hf.py
```

![image-20241111212837686](./assets/image-20241111212837686.png)

## 6.下载 NLTK 相关资源

```bash
cd /root
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
cd nltk_data
mv packages/*  ./
cd tokenizers
unzip punkt.zip
cd ../taggers
unzip averaged_perceptron_tagger.zip
```

![image-20241111213245372](./assets/image-20241111213245372.png)

## 7.是否使用 LlamaIndex 前后对比

### 不使用 LlamaIndex RAG（仅API）

新建`test_internlm.py`文件，并编辑代码

```bash
cd ~/llamaindex_demo
touch test_internlm.py
```

```python
from openai import OpenAI

base_url = "https://internlm-chat.intern-ai.org.cn/puyu/api/v1/"
api_key = "xxxxxxxxxxxxxxxxxxxx"
model="internlm2.5-latest"

# base_url = "https://api.siliconflow.cn/v1"
# api_key = "sk-请填写准确的 token！"
# model="internlm/internlm2_5-7b-chat"

client = OpenAI(
    api_key=api_key , 
    base_url=base_url,
)

chat_rsp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Jim.Fan的UID是多少？"}],
)

for choice in chat_rsp.choices:
    print(choice.message.content)
```

![image-20241111214659643](./assets/image-20241111214659643.png)

运行`test_internlm.py`查看结果

```bash
python test_internlm.py
```

![image-20241111214715578](./assets/image-20241111214715578.png)