import io
import sys
from pathlib import Path

import dataframe_image as dfi
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
import textwrap

DATA_START = "<<<DATA_START>>>"
DATA_END = "<<<DATA_END>>>"

# --- 1. 字体设置 ---
for font_name in ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Micro Hei", "Arial Unicode MS"):
    try:
        font_manager.findfont(font_name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [font_name]
        break
    except (ValueError, RuntimeError):
        continue
plt.rcParams["axes.unicode_minus"] = False

# --- 2. 数据准备（CSV 源） ---
default_raw = """<<<DATA_START>>>
资产名称,技术指标分析,推荐操作/观点
纳斯达克 100 / 标普 500,从高点回撤 5%-6%；跌破 60 日均线；处于中性调整水平,正常回调非崩盘；分批买入；等待趋势反转
黄金 / 白银 / 油气,11 月逆势上涨（如 GDX 涨 5%）,作为防御性资产配置；分散风险效果显著
英伟达 (NVDA),期权隐含波动大（预期 6-7% 波动）；Put/Call Ratio 偏牛 (0.56)；分析师目标价 >$200,财报博弈风险大（可能横盘杀期权）；建议长期持有；不建议赌财报方向
比特币 (BTC),反弹至 92000 美元附近；市场风险风向标,若反弹将带动科技股回暖；观察其动向判断大盘
特斯拉 (TSLA),跌破 60 日均线；RSI 超卖；关键支撑看 11 月中旬低点及 $350,长期上升趋势未变；勿因短期波动离场；关注支撑位
Meta,每日创新低（寻底中）；RSI 超卖；MACD 逐渐好转,分批建仓（左侧交易）；给予时间等待底部形成
<<<DATA_END>>>"""


def extract_csv_payload(raw_text: str) -> str:
    start_idx = raw_text.find(DATA_START)
    end_idx = raw_text.find(DATA_END, start_idx + len(DATA_START)) if start_idx != -1 else -1
    if start_idx != -1 and end_idx != -1:
        payload = raw_text[start_idx + len(DATA_START) : end_idx].strip()
        if payload:
            return payload
    return raw_text.strip()


csv_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("md_table.csv")
if csv_file.exists():
    raw_text = csv_file.read_text(encoding="utf-8")
else:
    raw_text = default_raw

csv_text = extract_csv_payload(raw_text)

df = pd.read_csv(io.StringIO(csv_text), sep=",")

# 【关键修复步骤 1】：强制将所有数据转换为字符串
# Matplotlib 引擎如果检测到数字（如 "0.96%"），会无视 CSS 强制右对齐。
# 转为 str 后，它就会听从 text-align: left 的指令了。
df = df.astype(str)

# --- 3. 文本换行处理 ---
def wrap_col(df: pd.DataFrame, width_map: dict[str, int]) -> pd.DataFrame:
    def wrap_val(val: str, width: int) -> str:
        # 确保处理的是字符串
        return "\n".join(textwrap.wrap(str(val), width=width, break_long_words=True))

    wrapped_df = df.copy()
    for col, width in width_map.items():
        if col in wrapped_df.columns:
            wrapped_df[col] = wrapped_df[col].map(lambda v: wrap_val(v, width))
    return wrapped_df

# 设置换行宽度（数字越小越窄）
col_widths = {
    df.columns[0]: 14,
    df.columns[1]: 22,
    df.columns[2]: 32 
}
df_wrapped = wrap_col(df, col_widths)

# --- 4. 样式设置 ---
base_styles = [
    dict(
        selector="table",
        props=[
            ("border-collapse", "collapse"),
            ("width", "100%"),
            ("background-color", "white"),
        ],
    ),
    # 【关键修复步骤 2】：针对表头(th)强制左对齐
    dict(
        selector="th",
        props=[
            ("background-color", "#f3f4f6"),
            ("color", "#1f2937"),
            ("font-weight", "bold"),
            ("padding", "12px 12px"),
            ("text-align", "left"),  # 表头左对齐
            ("border-bottom", "2px solid #000"),
        ],
    ),
    # 【关键修复步骤 3】：针对单元格(td)强制左对齐
    dict(
        selector="td",
        props=[
            ("padding", "10px 12px"),
            ("text-align", "left"),  # 单元格左对齐
            ("border-bottom", "1px solid #e5e7eb"),
            ("vertical-align", "top"),
        ],
    ),
    dict(
        selector="caption", 
        props=[
            ("caption-side", "top"), 
            ("font-size", "16pt"), 
            ("font-weight", "bold"), 
            ("padding", "10px"),
            ("color", "black"),
            ("text-align", "center") # 只有标题居中
        ]
    ),
]

# 应用样式
styled = df_wrapped.style.set_table_styles(base_styles)

# 【关键修复步骤 4】：使用 set_properties 再次全局覆盖
styled = styled.set_properties(**{
    "text-align": "left",        # 再次确认左对齐
    "white-space": "pre-wrap",   # 确保换行符生效
    "font-size": "12pt",
    "color": "#111827",
})

# 第一列加粗
styled = styled.set_properties(subset=[df.columns[0]], **{"font-weight": "bold"})

# 隐藏索引
if hasattr(styled, "hide_index"):
    styled = styled.hide_index()
else:
    styled = styled.hide(axis="index")

styled = styled.set_caption("资产观察列表")

# --- 5. 导出图片 ---
dfi.export(styled, "table_output_final.png", table_conversion="matplotlib", dpi=200)

print("图片已生成：table_output_final.png")
