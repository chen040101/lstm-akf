from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT = Path(__file__).resolve().parent
FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
S = 2


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size * S)

F_TITLE = font(34, True)
F_SUB = font(18)
F_HEAD = font(24, True)
F_NODE = font(20)
F_SMALL = font(17)
F_TINY = font(15)
F_FORMULA = font(25, True)

INK = "#111827"
MUTED = "#4b5563"
NAVY = "#17213f"
GRID = "#e5e7eb"
WHITE = "#ffffff"


def sc(v: float) -> int:
    return int(round(v * S))


class Canvas:
    def __init__(self, w: int, h: int, bg: str = "white") -> None:
        self.w = w
        self.h = h
        self.img = Image.new("RGB", (w * S, h * S), bg)
        self.d = ImageDraw.Draw(self.img)

    def save(self, path: Path) -> None:
        final = self.img.resize((self.w, self.h), Image.Resampling.LANCZOS)
        final.save(path, quality=98)

    def text_w(self, txt: str, fnt: ImageFont.FreeTypeFont) -> float:
        box = self.d.textbbox((0, 0), txt, font=fnt)
        return (box[2] - box[0]) / S

    def wrap(self, txt: str, fnt: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
        lines: list[str] = []
        for raw in txt.split("\n"):
            if not raw:
                lines.append("")
                continue
            buf = ""
            for ch in raw:
                test = buf + ch
                if not buf or self.text_w(test, fnt) <= max_w:
                    buf = test
                else:
                    lines.append(buf)
                    buf = ch
            if buf:
                lines.append(buf)
        return lines

    def text(self, x: float, y: float, txt: str, fnt: ImageFont.FreeTypeFont, fill: str = INK, anchor: str | None = None) -> None:
        self.d.text((sc(x), sc(y)), txt, font=fnt, fill=fill, anchor=anchor)

    def line(self, pts: list[tuple[float, float]], fill: str = INK, width: float = 2.0) -> None:
        self.d.line([(sc(x), sc(y)) for x, y in pts], fill=fill, width=sc(width), joint="curve")

    def rect(self, x: float, y: float, w: float, h: float, fill: str = WHITE, outline: str = NAVY, width: float = 2.0, radius: float = 10) -> None:
        self.d.rounded_rectangle((sc(x), sc(y), sc(x + w), sc(y + h)), radius=sc(radius), fill=fill, outline=outline, width=sc(width))

    def dashed_rect(self, x: float, y: float, w: float, h: float, outline: str = NAVY, width: float = 2.0, dash: float = 10, gap: float = 7, fill: str | None = None) -> None:
        if fill:
            self.d.rounded_rectangle((sc(x), sc(y), sc(x + w), sc(y + h)), radius=sc(12), fill=fill)
        self.dashed_line((x, y), (x + w, y), outline, width, dash, gap)
        self.dashed_line((x + w, y), (x + w, y + h), outline, width, dash, gap)
        self.dashed_line((x + w, y + h), (x, y + h), outline, width, dash, gap)
        self.dashed_line((x, y + h), (x, y), outline, width, dash, gap)

    def dashed_line(self, p1: tuple[float, float], p2: tuple[float, float], fill: str = INK, width: float = 2, dash: float = 10, gap: float = 7) -> None:
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux, uy = dx / length, dy / length
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            self.line([(x1 + ux * pos, y1 + uy * pos), (x1 + ux * end, y1 + uy * end)], fill, width)
            pos += dash + gap

    def arrow(
        self,
        pts: list[tuple[float, float]],
        fill_or_label: str = INK,
        width_or_label_pos: float | tuple[float, float] = 2.4,
        label: str | None = None,
        label_pos: tuple[float, float] | None = None,
        dashed: bool = False,
    ) -> None:
        # Supports both arrow(points, "label", (x, y)) and
        # arrow(points, "#color", 2.8, "label", (x, y)).
        if isinstance(width_or_label_pos, tuple):
            fill = INK
            width = 2.4
            label = fill_or_label
            label_pos = width_or_label_pos
        else:
            fill = fill_or_label
            width = width_or_label_pos
        if dashed:
            for a, b in zip(pts, pts[1:]):
                self.dashed_line(a, b, fill, width, 11, 8)
        else:
            self.line(pts, fill, width)
        x1, y1 = pts[-2]
        x2, y2 = pts[-1]
        ang = math.atan2(y2 - y1, x2 - x1)
        size = 12
        tri = [
            (x2, y2),
            (x2 - size * math.cos(ang - math.pi / 6), y2 - size * math.sin(ang - math.pi / 6)),
            (x2 - size * math.cos(ang + math.pi / 6), y2 - size * math.sin(ang + math.pi / 6)),
        ]
        self.d.polygon([(sc(x), sc(y)) for x, y in tri], fill=fill)
        if label:
            lx, ly = label_pos if label_pos else ((pts[0][0] + pts[-1][0]) / 2, (pts[0][1] + pts[-1][1]) / 2)
            lines = self.wrap(label, F_SMALL, 230)
            max_w = max((self.text_w(s, F_SMALL) for s in lines), default=0)
            lh = 22
            bx, by = lx - max_w / 2 - 7, ly - len(lines) * lh / 2 - 3
            self.d.rounded_rectangle((sc(bx), sc(by), sc(bx + max_w + 14), sc(by + len(lines) * lh + 6)), radius=sc(5), fill=WHITE)
            for i, line in enumerate(lines):
                self.text(lx, by + 4 + i * lh, line, F_SMALL, fill=INK, anchor="ma")

    def node(self, x: float, y: float, w: float, h: float, title: str, body: str = "", fill: str = WHITE, outline: str = NAVY, title_fill: str = NAVY, shadow: bool = False) -> None:
        if shadow:
            self.d.rounded_rectangle((sc(x + 5), sc(y + 6), sc(x + w + 5), sc(y + h + 6)), radius=sc(13), fill="#d1d5db")
        self.rect(x, y, w, h, fill, outline, 2.0, 12)
        title_lines = self.wrap(title, F_HEAD, w - 30)
        body_lines = self.wrap(body, F_NODE, w - 32) if body else []
        lh1, lh2 = 30, 27
        total = len(title_lines) * lh1 + (7 if body_lines else 0) + len(body_lines) * lh2
        yy = y + (h - total) / 2
        for line in title_lines:
            self.text(x + w / 2, yy, line, F_HEAD, fill=title_fill, anchor="ma")
            yy += lh1
        if body_lines:
            yy += 7
        for line in body_lines:
            self.text(x + w / 2, yy, line, F_NODE, fill=INK, anchor="ma")
            yy += lh2

    def small_card(self, x: float, y: float, w: float, h: float, text_: str, fill: str = "#f8fafc", outline: str = NAVY) -> None:
        self.rect(x, y, w, h, fill, outline, 1.4, 7)
        lines = self.wrap(text_, F_SMALL, w - 14)
        yy = y + (h - len(lines) * 22) / 2
        for line in lines:
            self.text(x + w / 2, yy, line, F_SMALL, INK, anchor="ma")
            yy += 22

    def plus(self, x: float, y: float, r: float = 17, fill: str = WHITE, outline: str = NAVY) -> None:
        self.d.ellipse((sc(x - r), sc(y - r), sc(x + r), sc(y + r)), fill=fill, outline=outline, width=sc(2))
        self.line([(x - 8, y), (x + 8, y)], outline, 2)
        self.line([(x, y - 8), (x, y + 8)], outline, 2)

    def times(self, x: float, y: float, r: float = 17, fill: str = WHITE, outline: str = NAVY) -> None:
        self.d.ellipse((sc(x - r), sc(y - r), sc(x + r), sc(y + r)), fill=fill, outline=outline, width=sc(2))
        self.line([(x - 7, y - 7), (x + 7, y + 7)], outline, 2)
        self.line([(x - 7, y + 7), (x + 7, y - 7)], outline, 2)

    def sequence(self, x: float, y: float, w: float, color: str = NAVY) -> None:
        self.rect(x, y, w, 86, "#ffffff", color, 2, 12)
        self.text(x + w / 2, y + 14, "历史中心点序列", F_HEAD, color, anchor="ma")
        for i in range(8):
            px = x + 42 + i * ((w - 84) / 7)
            py = y + 58 + 9 * math.sin(i / 7 * math.pi * 1.2)
            self.d.ellipse((sc(px - 5), sc(py - 5), sc(px + 5), sc(py + 5)), fill=color)
            if i > 0:
                prev_x = x + 42 + (i - 1) * ((w - 84) / 7)
                prev_y = y + 58 + 9 * math.sin((i - 1) / 7 * math.pi * 1.2)
                self.line([(prev_x, prev_y), (px, py)], color, 1.6)
        self.text(x + w / 2, y + 91, "history_xy [B,15,2]", F_SMALL, MUTED, anchor="ma")


def draw_academic() -> None:
    c = Canvas(1800, 1080, "white")
    c.text(900, 35, "LSTM-AKF 目标中心点多步预测网络结构", F_TITLE, NAVY, "ma")
    c.text(900, 76, "AKF 基线外推 + LSTM 双预测分支 + 自适应门控融合", F_SUB, MUTED, "ma")

    c.sequence(80, 130, 320, NAVY)
    c.node(530, 118, 340, 115, "LSTM Encoder", "2 layers, hidden=64\nhistory_embedding [B,64]", "#ffffff")
    c.node(80, 365, 350, 130, "AKF 基线生成", "状态 [cx, cy, vx, vy, ax, ay]\n外推未来 1~15 帧", "#ffffff")
    c.node(80, 535, 350, 90, "baseline_xy", "[B,15,2]", "#ffffff")

    c.node(550, 330, 300, 120, "Residual Head", "MLP: 64→128→64→30\npred_delta_xy [B,15,2]", "#ffffff")
    c.node(550, 570, 300, 120, "Direct Head", "MLP: 64→128→64→30\ndirect_offset_xy [B,15,2]", "#ffffff")
    c.node(930, 330, 300, 112, "残差候选轨迹", "residual_pred =\nbaseline + delta", "#ffffff")
    c.node(930, 570, 300, 112, "直推候选轨迹", "direct_pred =\nlast_xy + offset", "#ffffff")

    c.dashed_rect(1270, 265, 350, 390, NAVY, 2, 10, 7, None)
    c.text(1445, 286, "Gate Fusion", F_HEAD, NAVY, "ma")
    c.node(1300, 330, 290, 110, "拼接门控输入", "concat(h, residual, direct)\n[B,15,68]", "#ffffff")
    c.node(1300, 482, 290, 95, "Gate Head", "MLP + Sigmoid\ngate [B,15,1]", "#ffffff")

    c.node(1240, 760, 440, 108, "自适应融合", "pred = gate × residual_pred + (1-gate) × direct_pred", "#ffffff")
    c.node(720, 850, 360, 112, "最终输出", "pred_xy [B,15,2]\n未来 1~15 帧中心点", "#ffffff")

    c.arrow([(400, 174), (530, 174)], "history_xy", (465, 145))
    c.arrow([(240, 216), (240, 365)], "历史观测", (305, 290))
    c.arrow([(700, 233), (700, 330)])
    c.arrow([(700, 233), (700, 570)])
    c.arrow([(430, 580), (930, 386)], "baseline_xy", (615, 492))
    c.arrow([(850, 390), (930, 390)], "delta", (890, 365))
    c.arrow([(400, 174), (500, 174), (500, 630), (550, 630)], "last_xy", (500, 520))
    c.arrow([(850, 630), (930, 630)], "offset", (890, 606))
    c.arrow([(1230, 386), (1300, 386)])
    c.arrow([(1230, 626), (1300, 390)])
    c.arrow([(700, 233), (1230, 233), (1300, 386)], "repeat h", (1030, 214))
    c.arrow([(1445, 440), (1445, 482)])
    c.arrow([(1445, 577), (1445, 760)], "gate", (1505, 670))
    c.arrow([(1080, 442), (1080, 760), (1240, 812)], "residual_pred", (1140, 720))
    c.arrow([(1080, 682), (1080, 812), (1240, 812)], "direct_pred", (1140, 835))
    c.arrow([(1240, 814), (1080, 905)], "pred_xy", (1160, 880))
    c.text(75, 1016, "图注：虚线框表示门控融合模块；AKF 分支不引入额外神经网络输入，速度/加速度由状态估计得到。", F_SMALL, MUTED)
    c.save(OUT / "01_lstm_akf_academic.png")


def draw_color_modules() -> None:
    c = Canvas(1900, 1080, "#f7fbff")
    c.text(950, 35, "LSTM-AKF 多分支融合预测模型", F_TITLE, "#0f172a", "ma")
    c.text(950, 78, "面向装甲板中心点的未来 1~15 帧轨迹预测", F_SUB, "#475569", "ma")

    blue = "#dbeafe"; green = "#dcfce7"; amber = "#fef3c7"; orange = "#ffedd5"; rose = "#ffe4e6"; purple = "#ede9fe"; cyan = "#cffafe"; red = "#fecaca"
    c.sequence(70, 145, 330, "#2563eb")
    c.node(525, 130, 360, 128, "LSTM 时序编码器", "输入历史 xy，提取运动模式\n输出 h [B,64]", blue, "#2563eb", "#1d4ed8", True)
    c.node(70, 392, 340, 128, "AKF 状态估计器", "估计位置、速度、加速度\n生成物理先验基线", green, "#16a34a", "#166534", True)
    c.node(70, 560, 340, 88, "Baseline Trajectory", "baseline_xy [B,15,2]", green, "#16a34a", "#166534", True)

    c.node(520, 340, 310, 122, "残差预测头", "MLP 输出 Δxy\n学习修正 AKF 基线", amber, "#d97706", "#92400e", True)
    c.node(520, 600, 310, 122, "直接预测头", "MLP 输出 offset\n从 last_xy 直接外推", orange, "#ea580c", "#9a3412", True)

    c.plus(900, 410, 19, "#fff7ed", "#ea580c")
    c.plus(900, 670, 19, "#fff7ed", "#ea580c")
    c.node(965, 342, 300, 136, "残差候选", "baseline_xy + Δxy\nresidual_pred_xy", amber, "#d97706", "#92400e", True)
    c.node(965, 602, 300, 136, "直推候选", "last_xy + offset\ndirect_pred_xy", orange, "#ea580c", "#9a3412", True)

    c.node(1345, 225, 350, 160, "门控权重学习", "concat(h, residual, direct)\nGate Head + Sigmoid\ngate [B,15,1]", purple, "#7c3aed", "#5b21b6", True)
    c.node(1350, 542, 345, 155, "自适应融合层", "gate × residual\n+ (1-gate) × direct", rose, "#e11d48", "#9f1239", True)
    c.node(1510, 815, 300, 135, "输出端", "pred_xy [B,15,2]\n未来轨迹点", cyan, "#0891b2", "#155e75", True)

    # decorative future trajectory
    for i in range(9):
        px = 1590 + i * 19
        py = 780 - 34 * math.sin(i / 8 * math.pi)
        c.d.ellipse((sc(px - 5), sc(py - 5), sc(px + 5), sc(py + 5)), fill="#0891b2")
        if i > 0:
            px0 = 1590 + (i - 1) * 19
            py0 = 780 - 34 * math.sin((i - 1) / 8 * math.pi)
            c.line([(px0, py0), (px, py)], "#0891b2", 1.8)
    c.text(1688, 748, "t+1 ... t+15", F_TINY, "#155e75")

    arrow_color = "#1e3a8a"
    c.arrow([(400, 188), (525, 188)], arrow_color, 2.8, "history_xy", (462, 158))
    c.arrow([(235, 236), (235, 392)], arrow_color, 2.8, "历史点", (292, 315))
    c.arrow([(705, 258), (705, 340)], arrow_color, 2.8)
    c.arrow([(705, 258), (705, 600)], arrow_color, 2.8)
    c.arrow([(410, 604), (881, 410)], arrow_color, 2.8, "baseline", (610, 500))
    c.arrow([(830, 402), (881, 408)], arrow_color, 2.8, "Δxy", (858, 374))
    c.arrow([(900, 410), (965, 410)], arrow_color, 2.8)
    c.arrow([(400, 188), (470, 188), (470, 670), (881, 670)], arrow_color, 2.8, "last_xy", (460, 520))
    c.arrow([(830, 662), (881, 668)], arrow_color, 2.8, "offset", (858, 636))
    c.arrow([(900, 670), (965, 670)], arrow_color, 2.8)
    c.arrow([(1265, 410), (1345, 302)], arrow_color, 2.8, "residual", (1322, 394))
    c.arrow([(1265, 670), (1345, 310)], arrow_color, 2.8, "direct", (1310, 548))
    c.arrow([(885, 195), (1180, 195), (1345, 303)], arrow_color, 2.8, "h", (1110, 172))
    c.arrow([(1518, 385), (1518, 542)], arrow_color, 2.8, "gate", (1570, 466))
    c.arrow([(1265, 410), (1350, 610)], arrow_color, 2.8)
    c.arrow([(1265, 670), (1350, 630)], arrow_color, 2.8)
    c.arrow([(1522, 697), (1605, 815)], arrow_color, 2.8, "pred_xy", (1590, 742))

    c.text(75, 1016, "说明：彩色模块图适合答辩 PPT；颜色区分 AKF 物理先验、LSTM 数据驱动预测与门控融合。", F_SMALL, "#475569")
    c.save(OUT / "02_lstm_akf_color_modules.png")


def draw_mechanism() -> None:
    c = Canvas(1800, 1080, "#fbfbfb")
    c.text(900, 35, "LSTM-AKF 门控融合机制示意图", F_TITLE, NAVY, "ma")
    c.text(900, 76, "用可学习 gate 在 AKF 残差预测与 LSTM 直推预测之间自适应分配权重", F_SUB, MUTED, "ma")

    # group boxes
    c.dashed_rect(65, 125, 440, 785, "#6b7280", 2, 10, 7, "#ffffff")
    c.text(285, 148, "输入与 AKF 先验", F_HEAD, NAVY, "ma")
    c.dashed_rect(540, 125, 520, 785, "#6b7280", 2, 10, 7, "#ffffff")
    c.text(800, 148, "LSTM 预测候选", F_HEAD, NAVY, "ma")
    c.dashed_rect(1095, 125, 640, 785, "#6b7280", 2, 10, 7, "#ffffff")
    c.text(1415, 148, "门控融合", F_HEAD, NAVY, "ma")

    c.sequence(120, 210, 330, NAVY)
    c.node(120, 390, 330, 120, "AKF 状态空间", "s=[cx,cy,vx,vy,ax,ay]\nKalman update / predict", "#ffffff")
    c.node(120, 590, 330, 95, "基线外推", "Y_AKF = baseline_xy", "#ffffff")
    c.node(610, 210, 370, 125, "LSTM Encoder", "H = LSTM(history_xy)\nH ∈ R^{B×64}", "#ffffff")
    c.node(610, 430, 370, 120, "Residual Head", "ΔY = MLP_res(H)\nY_res = Y_AKF + ΔY", "#ffffff")
    c.node(610, 650, 370, 120, "Direct Head", "O = MLP_dir(H)\nY_dir = last_xy + O", "#ffffff")

    c.node(1160, 245, 500, 120, "Gate 输入", "Z = concat(repeat(H), Y_res, Y_dir)\nZ ∈ R^{B×15×68}", "#ffffff")
    c.node(1160, 430, 500, 110, "Gate Head", "G = sigmoid(MLP_gate(Z))\nG ∈ R^{B×15×1}", "#ffffff")
    c.node(1160, 625, 500, 150, "融合公式", "Y_pred = G ⊙ Y_res + (1-G) ⊙ Y_dir", "#ffffff")
    c.node(1280, 820, 260, 75, "预测输出", "Y_pred [B,15,2]", "#ffffff")

    c.arrow([(285, 296), (285, 390)], "观测序列", (350, 345))
    c.arrow([(285, 510), (285, 590)], "predict", (335, 550))
    c.arrow([(450, 637), (510, 637), (510, 490), (610, 490)], "Y_AKF", (515, 585))
    c.arrow([(450, 255), (610, 270)], "history", (528, 230))
    c.arrow([(795, 335), (795, 430)])
    c.arrow([(760, 335), (760, 375), (575, 375), (575, 710), (610, 710)])
    c.arrow([(980, 490), (1160, 305)], "Y_res", (1075, 370))
    c.arrow([(980, 710), (1160, 320)], "Y_dir", (1085, 555))
    c.arrow([(980, 272), (1160, 305)], "repeat(H)", (1070, 250))
    c.arrow([(1410, 365), (1410, 430)])
    c.arrow([(1410, 540), (1410, 625)], "G", (1455, 580))
    c.arrow([(980, 490), (1085, 575), (1160, 665)], "Y_res", (1060, 585))
    c.arrow([(980, 710), (1085, 735), (1160, 705)], "Y_dir", (1060, 750))
    c.arrow([(1410, 775), (1410, 820)])
    c.text(95, 985, "图注：G 为逐未来时间步学习得到的融合权重；当 AKF 基线可靠时增大残差分支权重，反之更多依赖直接时序外推。", F_SMALL, MUTED)
    c.save(OUT / "03_lstm_akf_gate_mechanism.png")


def main() -> None:
    draw_academic()
    draw_color_modules()
    draw_mechanism()
    print(OUT / "01_lstm_akf_academic.png")
    print(OUT / "02_lstm_akf_color_modules.png")
    print(OUT / "03_lstm_akf_gate_mechanism.png")


if __name__ == "__main__":
    main()
