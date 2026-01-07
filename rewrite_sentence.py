import pandas as pd
from tqdm import tqdm
import dashscope
from http import HTTPStatus
import time
import glob

# ================= 配置区 =================
dashscope.api_key = "sk-825640d75c444c9baf2bbda6d8161e52"

INPUT_FILE = "data/test.csv"
OUTPUT_FILE = "data/test_sentence.csv"

TARGET_TOTAL = 500      # 只处理前 500 条
CHUNK_SIZE = 50         # 每 50 条落盘一次
# ==========================================

def rewrite_dialogue_sentence_level(text):
    """
    使用大模型进行“整句级别改写”
    强调句式、结构调整，而非仅词替换
    """
    try:
        response = dashscope.Generation.call(
            model="qwen-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "你是一名自然语言处理研究助手。"
                },
                {
                    "role": "user",
                    "content": f"""
请对下面的客服对话进行【整句级别的语义保持改写】，要求：

1. 保持原始对话的真实意图和欺诈/非欺诈属性不变
2. 不改变任何事实含义
3. 尽量通过【句式调整、语序变化、信息重组】来改写整句话
4. 避免仅做简单的同义词替换
5. 保持 left / right 的对话结构不变

请只输出改写后的完整对话文本，不要输出任何解释或说明。

原始对话：
{text}
"""
                }
            ],
            result_format="message",
            temperature=0.7
        )

        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0].message["content"]
            return content.replace("```", "").strip()
        else:
            return text

    except Exception:
        return text


def main():
    df = pd.read_csv(INPUT_FILE)

    # ========= 断点续跑检测 =========
    existing_parts = glob.glob("data/test_sentence_part_*.csv")
    if existing_parts:
        indices = []
        for f in existing_parts:
            try:
                indices.append(int(f.split("_")[-1].split(".")[0]))
            except:
                continue
        processed_count = max(indices) if indices else 0
        print(f"检测到历史进度，将从第 {processed_count + 1} 条继续处理...")
    else:
        processed_count = 0
        print("未检测到历史分段，从头开始运行...")

    # ========= 分段处理 =========
    for chunk_start in range(processed_count, TARGET_TOTAL, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, TARGET_TOTAL)
        print(f"\n正在处理分段: {chunk_start + 1} - {chunk_end} / {TARGET_TOTAL}")

        rewritten_texts = []
        for i in tqdm(range(chunk_start, chunk_end)):
            raw_text = df.loc[i, "specific_dialogue_content"]
            rewritten_texts.append(
                rewrite_dialogue_sentence_level(raw_text)
            )

        chunk_df = df.iloc[chunk_start:chunk_end].copy()
        chunk_df["specific_dialogue_content"] = rewritten_texts

        chunk_df.to_csv(
            f"data/test_sentence_part_{chunk_start+1}_{chunk_end}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print(f"分段 {chunk_start+1}-{chunk_end} 已保存，休息 3 秒...")
        time.sleep(3)

    # ========= 自动合并 =========
    print("\n正在合并所有分段文件...")
    all_part_files = glob.glob("data/test_sentence_part_*.csv")
    all_part_files.sort(key=lambda x: int(x.split("_")[-2]))

    dfs = [pd.read_csv(f) for f in all_part_files]

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        final_df = final_df.head(TARGET_TOTAL)
        final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"整句改写数据集生成完成：{OUTPUT_FILE}（共 {len(final_df)} 条）")
    else:
        print("未发现可合并的分段文件。")


if __name__ == "__main__":
    main()
