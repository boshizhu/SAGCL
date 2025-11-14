import torch
from torch_geometric.data import Data  # 导入以进行类型检查（可选）

# 替换为您的.pt文件路径
graph_data = torch.load(f"D:\LLM\LLMNodeBed-main\LLMNodeBed-main\datasets\\cora.pt",
                                    weights_only=False).to("cpu")
data = graph_data

print(data.y)
# 检查加载的数据类型（可选，用于调试）
print(f"Loaded data type: {type(data)}")
if isinstance(data, dict):
    print("Keys in data:", list(data.keys()))  # 如果是字典，打印键以帮助调试

# 提取节点的raw text列表
# 假设存储在 data.raw_text 中（调整为实际键名，如 data.node_texts 或 data.texts）
# 如果是张量，需要转换为列表：texts = data.raw_text.tolist()
texts = data.raw_texts  # 调整这里！

# 如果texts不是列表或张量，添加检查
if not (isinstance(texts, list) or torch.is_tensor(texts)):
    raise ValueError("Raw text data is not a list or tensor. Please check the attribute name and adjust the code.")

# 如果是张量，转换为列表
if torch.is_tensor(texts):
    texts = texts.tolist()

# 计算每个节点的文本长度（过滤非字符串项）
lengths = [len(text) for text in texts if isinstance(text, str)]

# 处理边缘情况：如果没有有效文本
if not lengths:
    print("No valid text data found in the file.")
else:
    max_length = max(lengths)
    min_length = min(lengths)
    avg_length = sum(lengths) / len(lengths) if lengths else 0

    # 打印结果
    print(f"Number of nodes with text: {len(lengths)}")
    print(f"Maximum text length: {max_length}")
    print(f"Minimum text length: {min_length}")
    print(f"Average text length: {avg_length:.2f}")  # 保留两位小数