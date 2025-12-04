import argparse
import io
import logging
import json
import tempfile
import textwrap
from pathlib import Path

import dataframe_image as dfi
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm
import ast

DATA_START = "<<<DATA_START>>>"
DATA_END = "<<<DATA_END>>>"

FONT_CANDIDATES = ("Noto Sans CJK SC", "Microsoft YaHei", "SimHei", "PingFang SC", "WenQuanYi Micro Hei", "Arial Unicode MS")

# 更新默认数据为 JSON 格式，方便测试
DEFAULT_TABLE_RAW = """<<<DATA_START>>>
[["资产名称", "技术指标分析", "推荐操作/观点"],["UnitedHealth (UNH)", "估值：基于2025年EPS 16.25和20倍P/E，短期公允估值约$320。该股是价值股，非成长 爆发股。", "维持长线持有观点，预计明年有望站上$400。新仓安全建仓区间为$320-$330。该股具有防御性，适合追求稳健的投资者。"],["META Platforms (META)", "P/E Ratio: 26 (低于历史平均28-29)。技术面：$650-$620是小幅建仓的区域；$620以下风险收益比更佳。", "股价下跌主要由一次性非现金税务费用和提升AI基建的 资本开支引起，基本面强劲。当前估值具吸引力，可开始在建议区间内分批建仓。"],["Rubrik (RBRK)", "PS Ratio: 14 (低于同业CRWD 28，PANW 16)。目前处于亏损 状态，2026年预计盈利。理想建仓空间为$60-$70。", "高风险、高回报但确定性较强的高成长股，处于数据安全和备份赛道的中心。管理层预计现金流将转正，长期逻 辑坚固，建议可建小仓位并长期持有。"]]
<<<DATA_END>>>"""


def _init_table_font() -> str:
    """初始化字体并返回找到的最佳字体名称"""
    import matplotlib.font_manager as fm
    import logging

    system_fonts = {f.name for f in fm.fontManager.ttflist}
    logging.info(f"系统字体总数: {len(system_fonts)}")

    PREFERRED_FONTS = [
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "WenQuanYi Micro Hei"
    ]

    def _apply_font(font_name: str, strategy_name: str):
        # 1. 设置无衬线字体 (主字体)
        plt.rcParams["font.sans-serif"] = [font_name]
        plt.rcParams["font.family"] = "sans-serif"

        # 2. 【关键】强制设置衬线字体 (rm/serif) 也为该中文字体
        # 防止 Matplotlib 遇到标点符号或特定字符回退到默认不支持中文的字体
        plt.rcParams["font.serif"] = [font_name]

        # 3. 【关键】禁止数学公式模式切换字体
        # 防止遇到 $ % 等符号时切换到 STIX 导致乱码
        plt.rcParams['mathtext.default'] = 'regular'

        # 4. 解决负号显示
        plt.rcParams["axes.unicode_minus"] = False

        logging.info(f"{strategy_name}: 使用 {font_name}")

    # 策略 A: 精确匹配
    for font_name in PREFERRED_FONTS:
        if font_name in system_fonts:
            _apply_font(font_name, "策略A - 完美匹配")
            return font_name

    # 策略 B: 模糊匹配 (针对 Docker 环境)
    sans_cjk = [f for f in system_fonts if "Noto Sans CJK" in f]
    if sans_cjk:
        chosen_font = sorted(sans_cjk)[0]
        _apply_font(chosen_font, "策略B - 模糊匹配 (Sans优先)")
        return chosen_font

    # 策略 C: 保底
    any_cjk = [f for f in system_fonts if "CJK" in f]
    if any_cjk:
        chosen_font = any_cjk[0]
        _apply_font(chosen_font, "策略C - 最后保底")
        return chosen_font

    logging.error("严重错误：未找到任何支持中文的字体！")
    return "sans-serif"


def _wrap_columns(df: pd.DataFrame, width_map: dict[str, int]) -> pd.DataFrame:
    def wrap_val(val: str, width: int) -> str:
        return "\n".join(textwrap.wrap(str(val), width=width, break_long_words=True))

    wrapped_df = df.copy()
    for col, width in width_map.items():
        if col in wrapped_df.columns:
            wrapped_df[col] = wrapped_df[col].map(lambda v: wrap_val(v, width))
    return wrapped_df


def _extract_table_payload(raw_text: str) -> str:
    start_idx = raw_text.find(DATA_START)
    end_idx = raw_text.find(DATA_END, start_idx + len(DATA_START)) if start_idx != -1 else -1
    if start_idx != -1 and end_idx != -1:
        payload = raw_text[start_idx + len(DATA_START) : end_idx].strip()
        if payload:
            return payload
    return raw_text.strip()


def extract_table_payload(raw_text: str) -> str:
    """Public wrapper to extract content between placeholders."""
    payload = _extract_table_payload(raw_text)
    logging.info("Extracted table payload: %s", payload)
    return payload


def render_md_table_to_png(raw_payload: str, output_path: str, caption: str | None = None) -> str:
    # 1. 获取字体名称
    font_name = _init_table_font()

    # 2. 提取有效载荷
    payload_text = _extract_table_payload(raw_payload).strip()

    df = pd.DataFrame()
    used_parser = "unknown"

    # 3. 智能解析：优先 JSON List of Lists，失败则回退 CSV
    try:
        # --- 清洗 Markdown 标记 (如 ```json ... ```) ---
        clean_text = payload_text
        if clean_text.startswith("```"):
            lines = clean_text.split('\n')
            # 去掉第一行(可能含 ```json) 和 最后一行(可能含 ```)
            if "```" in lines[0]: lines = lines[1:]
            if len(lines) > 0 and "```" in lines[-1]: lines = lines[:-1]
            clean_text = "\n".join(lines).strip()

        # --- 尝试 JSON 解析 ---
        data = json.loads(clean_text)

        # 校验结构：必须是 [[头], [行], [行]]
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            headers = data[0]
            rows = data[1:]
            df = pd.DataFrame(rows, columns=headers)
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # 兼容对象列表 [{"A":1}, {"A":2}]
            df = pd.DataFrame(data)
        else:
            raise ValueError("JSON format not recognized as table")
        used_parser = "json"

    except (json.JSONDecodeError, ValueError) as e:
        # --- JSON 失败，尝试 Python literal，再失败回退到 CSV 解析 ---
        try:
            data = ast.literal_eval(payload_text)
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                headers = data[0]
                rows = data[1:]
                df = pd.DataFrame(rows, columns=headers)
                used_parser = "literal_eval"
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                df = pd.DataFrame(data)
                used_parser = "literal_eval"
            else:
                raise ValueError("literal_eval format not recognized as table")
        except Exception:
            logging.warning(f"结构化解析失败 ({e})，尝试 CSV 回退模式...")
            try:
                df = pd.read_csv(io.StringIO(payload_text), sep=",", engine="python")
                used_parser = "csv"
            except Exception as csv_e:
                logging.error(f"CSV 解析也失败: {csv_e}")
                raise ValueError("无法解析数据，请检查 Prompt 输出格式")

    # 4. 数据预处理：转为字符串，处理空值
    # 这一步很重要，因为 JSON 可能包含 null/None
    df = df.astype(str).replace("nan", "").replace("None", "")
    logging.info("表格解析方式=%s, 维度=%s", used_parser, df.shape)

    # 5. 表格排版与自动换行
    # 根据列数动态调整宽度，防止太宽或太窄
    # 这里简单假设前几列比较重要
    col_widths = {}
    if len(df.columns) > 0: col_widths[df.columns[0]] = 14
    if len(df.columns) > 1: col_widths[df.columns[1]] = 20
    if len(df.columns) > 2: col_widths[df.columns[2]] = 20

    df_wrapped = _wrap_columns(df, col_widths)

    # 6. CSS 样式注入
    # 直接使用字体名称，不加引号和 fallback，Matplotlib 更稳定
    css_font_family = font_name

    base_styles = [
        dict(
            selector="table",
            props=[
                ("border-collapse", "collapse"),
                ("width", "100%"),
                ("background-color", "white"),
                ("font-family", css_font_family),
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
                ("font-family", css_font_family),
            ],
        ),
        dict(
            selector="td",
            props=[
                ("padding", "14px 12px"),
                ("text-align", "left"),
                ("border-bottom", "1px solid #e5e7eb"),
                ("vertical-align", "top"),
                ("font-family", css_font_family),
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
                ("font-family", css_font_family),
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
            "font-family": css_font_family, # 再次兜底
        }
    )
    # 第一列加粗
    if len(df.columns) > 0:
        styled = styled.set_properties(subset=[df.columns[0]], **{"font-weight": "bold"})

    if hasattr(styled, "hide_index"):
        styled = styled.hide_index()
    else:
        styled = styled.hide(axis="index")

    styled = styled.set_caption(caption or "资产观察列表")

    dfi.export(styled, output_path, table_conversion="matplotlib", dpi=200)
    return output_path


def generate_table_image_file(raw_table: str | None = None, caption: str | None = None) -> str:
    table_content = (raw_table or DEFAULT_TABLE_RAW).strip()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        output_path = tmp.name
    try:
        return render_md_table_to_png(table_content, output_path, caption=caption)
    except Exception:
        logging.exception("渲染表格图片失败")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Render JSON/CSV to PNG.")
    parser.add_argument("--input", "-i", help="Input text file; defaults to built-in sample.")
    parser.add_argument("--output", "-o", help="Output PNG path", default="table_output_final.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.input:
        raw_content = Path(args.input).read_text(encoding="utf-8")
    else:
        raw_content = DEFAULT_TABLE_RAW

    output_path = Path(args.output)
    render_md_table_to_png(raw_content, str(output_path))
    print(f"图片已生成: {output_path}")


if __name__ == "__main__":
    main()
