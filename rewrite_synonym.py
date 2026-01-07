import pandas as pd
import random

SYNONYM_DICT = {
    "客服": ["客户服务", "服务人员", "客服人员", "专员"],
    "客服专员": ["客服人员", "服务专员", "客户顾问"],
    "客户经理": ["客户代表", "业务经理", "客户顾问"],
    "贷款专员": ["信贷专员", "融资顾问"],
    "客服中心": ["服务中心", "客户服务中心"],
    "工作人员": ["服务人员", "相关人员"],
    "银行": ["金融机构", "银行机构"],
    "工商银行": ["工行"],
    "农业银行": ["农行"],
    "建设银行": ["建行"],
    "招商银行": ["招行"],
    "支付宝": ["支付平台"],
    "京东": ["电商平台"],
    "淘宝": ["购物平台"],
    "快递": ["物流", "配送"],
    "点击": ["打开", "点开", "访问"],
    "下载": ["安装", "获取"],
    "填写": ["提交", "录入"],
    "提供": ["提交", "给出"],
    "操作": ["处理", "办理"],
    "处理": ["解决", "处置"],
    "验证": ["核实", "确认"],
    "确认": ["核对", "核实"],
    "申请": ["办理", "发起"],
    "链接": ["网址", "页面入口", "访问地址"],
    "验证码": ["校验码", "短信码", "验证代码"],
    "银行卡": ["银行账户", "卡号"],
    "账户": ["账号", "账户信息"],
    "身份信息": ["个人信息", "身份资料"],
    "订单": ["购买记录", "交易记录"],
    "退款": ["返款", "退费"],
    "赔偿": ["补偿", "理赔"],
    "转账": ["汇款", "打款"],
    "放款": ["发放资金", "到账"],
    "贷款": ["借款", "信贷"],
    "利息": ["利率", "息费"],
    "收益": ["回报", "收益率"],
    "别担心": ["不用担心", "请放心"],
    "非常安全": ["绝对安全", "安全可靠"],
    "非常简单": ["很简单", "操作不复杂"],
    "尽快": ["尽早", "及时"],
    "马上": ["立刻", "立即"]
}

def synonym_replace(text, replace_prob=0.3):
    candidates = [w for w in SYNONYM_DICT if w in text]
    if not candidates or random.random() > replace_prob:
        return text

    word = random.choice(candidates)
    return text.replace(word, random.choice(SYNONYM_DICT[word]), 1)

def main():
    input_path = "data/test.csv"
    output_path = "data/test_synonym.csv"

    df = pd.read_csv(input_path)

    df["specific_dialogue_content"] = (
        df["specific_dialogue_content"]
        .astype(str)
        .apply(synonym_replace)
    )

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print("同义词替换数据集已生成：", output_path)

if __name__ == "__main__":
    main()
