import pandas as pd
from tqdm import tqdm
import dashscope
from http import HTTPStatus
import os

# 配置通义千问 API Key
dashscope.api_key = "sk-825640d75c444c9baf2bbda6d8161e52"

# 加载完整数据
df = pd.read_csv("data/test.csv")

def rewrite_dialogue(text):
    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=[
            {
                "role": "system", 
                "content": "你是一名自然语言处理研究助手。"
            },
            {
                "role": "user", 
                "content": f"""
请对下面的客服对话进行语义保持的改写，要求：
1. 保持对话的原始意图不变
2. 不改变事实含义
3. 尽量替换表达方式、句式或关键词
4. 保持 left / right 对话结构

请只输出改写后的完整对话文本，不要输出任何解释。

原始对话：
{text}
"""
            }
        ],
        result_format='message',
        temperature=0.7
    )

    if response.status_code == HTTPStatus.OK:
        # 简单清洗，确保没有 Markdown 标签干扰 CSV
        content = response.output.choices[0].message['content']
        return content.replace("```", "").strip()
    else:
        raise Exception(f"Error Code: {response.code}, Message: {response.message}")

# 开始全量处理
adv_texts = []
output_path = "data/test_adv.csv"

print(f"开始全量处理 {len(df)} 条数据，预计耗时约 {len(df) * 3 / 60:.1f} 分钟...")

for i, text in enumerate(tqdm(df["specific_dialogue_content"])):
    try:
        new_text = rewrite_dialogue(text)
        adv_texts.append(new_text)
    except Exception as e:
        print(f"\n第 {i+1} 条数据调用失败，保留原值。错误信息: {e}")
        adv_texts.append(text)
    
    # 【安全机制】每 50 条保存一次临时文件，防止程序崩溃
    if (i + 1) % 50 == 0:
        temp_df = df.head(len(adv_texts)).copy()
        temp_df["specific_dialogue_content"] = adv_texts
        temp_df.to_csv(output_path, index=False, encoding="utf-8-sig")

# 最终保存
df["specific_dialogue_content"] = adv_texts
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\n[任务完成] 全量对抗数据集已生成：{output_path}")
