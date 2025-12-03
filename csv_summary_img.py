import argparse
import io
import logging
import tempfile
import textwrap
from pathlib import Path

import dataframe_image as dfi
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm

DATA_START = "<<<DATA_START>>>"
DATA_END = "<<<DATA_END>>>"

FONT_CANDIDATES = ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Micro Hei", "Arial Unicode MS")

DEFAULT_CSV_RAW = """<<<DATA_START>>>
资产名称,技术指标分析,推荐操作/观点
纳斯达克 100 / 标普 500,从高点回撤 5%-6%；跌破 60 日均线；处于中性调整水平,正常回调非崩盘；分批买入；等待趋势反转
黄金 / 白银 / 油气,11 月逆势上涨（如 GDX 涨 5%）,作为防御性资产配置；分散风险效果显著
英伟达 (NVDA),期权隐含波动大（预期 6-7% 波动）；Put/Call Ratio 偏牛 (0.56)；分析师目标价 >$200,财报博弈风险大（可能横盘杀期权）；建议长期持有；不建议赌财报方向
比特币 (BTC),反弹至 92000 美元附近；市场风险风向标,若反弹将带动科技股回暖；观察其动向判断大盘
特斯拉 (TSLA),跌破 60 日均线；RSI 超卖；关键支撑看 11 月中旬低点及 $350,长期上升趋势未变；勿因短期波动离场；关注支撑位
Meta,每日创新低（寻底中）；RSI 超卖；MACD 逐渐好转,分批建仓（左侧交易）；给予时间等待底部形成
<<<DATA_END>>>"""


def _init_table_font() -> None:
    import matplotlib.font_manager as fm
    import logging

    # 1. 获取系统所有字体名称
    system_fonts = {f.name for f in fm.fontManager.ttflist}
    logging.info(f"系统字体总数: {len(system_fonts)}")
    
    # 定义首选列表 (SC = 简体中文)
    PREFERRED_FONTS = [
        "Noto Sans CJK SC", 
        "Microsoft YaHei", 
        "SimHei", 
        "PingFang SC",
        "WenQuanYi Micro Hei"
    ]

    font_found = False
    
    # 策略 A: 尝试精确匹配首选字体
    for font_name in PREFERRED_FONTS:
        if font_name in system_fonts:
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["font.family"] = "sans-serif"
            logging.info(f"策略A - 完美匹配: 使用 {font_name}")
            font_found = True
            break
    
    # 策略 B: 如果没找到，寻找任何包含 "Noto Sans CJK" 的字体 (包含 JP/KR 等)
    # 这步是关键！它会优先选 Noto Sans (黑体) 而不是 Noto Serif (宋体)
    if not font_found:
        sans_cjk = [f for f in system_fonts if "Noto Sans CJK" in f]
        if sans_cjk:
            # 排序一下，通常把 SC 排前面（如果有的话），或者至少选一个 Sans
            chosen_font = sorted(sans_cjk)[0] 
            plt.rcParams["font.sans-serif"] = [chosen_font]
            plt.rcParams["font.family"] = "sans-serif"
            logging.info(f"策略B - 模糊匹配 (Sans优先): 使用 {chosen_font}")
            font_found = True

    # 策略 C: 实在不行，才用任意 CJK 字体保底 (这时才允许出现宋体)
    if not font_found:
        any_cjk = [f for f in system_fonts if "CJK" in f]
        if any_cjk:
            chosen_font = any_cjk[0]
            plt.rcParams["font.sans-serif"] = [chosen_font]
            plt.rcParams["font.family"] = "sans-serif"
            logging.warning(f"策略C - 最后保底 (可能是宋体): 使用 {chosen_font}")
            font_found = True

    if not font_found:
        logging.error("严重错误：未找到任何支持中文的字体！")

    # 解决负号显示问题
    plt.rcParams["axes.unicode_minus"] = False


def _wrap_columns(df: pd.DataFrame, width_map: dict[str, int]) -> pd.DataFrame:
    def wrap_val(val: str, width: int) -> str:
        return "\n".join(textwrap.wrap(str(val), width=width, break_long_words=True))

    wrapped_df = df.copy()
    for col, width in width_map.items():
        if col in wrapped_df.columns:
            wrapped_df[col] = wrapped_df[col].map(lambda v: wrap_val(v, width))
    return wrapped_df


def _extract_csv_payload(raw_text: str) -> str:
    start_idx = raw_text.find(DATA_START)
    end_idx = raw_text.find(DATA_END, start_idx + len(DATA_START)) if start_idx != -1 else -1
    if start_idx != -1 and end_idx != -1:
        payload = raw_text[start_idx + len(DATA_START) : end_idx].strip()
        if payload:
            return payload
    return raw_text.strip()


def extract_csv_payload(raw_text: str) -> str:
    """Public wrapper to extract CSV content between placeholders."""
    return _extract_csv_payload(raw_text)


def render_md_table_to_png(raw_csv: str, output_path: str) -> str:
    _init_table_font()

    csv_text = _extract_csv_payload(raw_csv)
    df = pd.read_csv(io.StringIO(csv_text), sep=",")
    df = df.astype(str)

    col_widths = {
        df.columns[0]: 14,
        df.columns[1]: 20,
        df.columns[2]: 20,
    }
    df_wrapped = _wrap_columns(df, col_widths)

    base_styles = [
        dict(
            selector="table",
            props=[
                ("border-collapse", "collapse"),
                ("width", "100%"),
                ("background-color", "white"),
            ],
        ),
        dict(
            selector="th",
            props=[
                ("background-color", "#f3f4f6"),
                ("color", "#1f2937"),
                ("font-weight", "bold"),
                ("padding", "12px 12px"),
                ("text-align", "left"),
                ("border-bottom", "2px solid #000"),
            ],
        ),
        dict(
            selector="td",
            props=[
                ("padding", "14px 12px"),
                ("text-align", "left"),
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
                ("text-align", "center"),
            ],
        ),
    ]

    styled = df_wrapped.style.set_table_styles(base_styles)
    styled = styled.set_properties(
        **{
            "text-align": "left",
            "white-space": "pre-wrap",
            "font-size": "12pt",
            "color": "#111827",
            "line-height": "1.6",
        }
    )
    styled = styled.set_properties(subset=[df.columns[0]], **{"font-weight": "bold"})

    if hasattr(styled, "hide_index"):
        styled = styled.hide_index()
    else:
        styled = styled.hide(axis="index")

    styled = styled.set_caption("资产观察列表")

    dfi.export(styled, output_path, table_conversion="matplotlib", dpi=200)
    return output_path


def generate_table_image_file(raw_csv: str | None = None) -> str:
    table_content = (raw_csv or DEFAULT_CSV_RAW).strip()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        output_path = tmp.name
    try:
        return render_md_table_to_png(table_content, output_path)
    except Exception:
        logging.exception("渲染 CSV 表格图片失败")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CSV (with <<<DATA_START/END>>> placeholder) to PNG.")
    parser.add_argument("--input", "-i", help="CSV text file containing placeholders; defaults to built-in sample.")
    parser.add_argument("--output", "-o", help="Output PNG path", default="table_output_final.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.input:
        raw_csv = Path(args.input).read_text(encoding="utf-8")
    else:
        raw_csv = DEFAULT_CSV_RAW

    output_path = Path(args.output)
    render_md_table_to_png(raw_csv, str(output_path))
    print(f"图片已生成: {output_path}")


if __name__ == "__main__":
    main()
