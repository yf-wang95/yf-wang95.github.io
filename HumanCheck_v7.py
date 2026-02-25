# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QInputDialog, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class ECGCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(12, 10), facecolor='#121212', dpi=100)
        self.fig.subplots_adjust(
            hspace=0.25,
            left=0.06,
            right=0.98,
            top=0.96,
            bottom=0.05
        )
        super().__init__(self.fig)
        self.axes = []

    def draw_record(self, record):
        num = record.n_sig
        sig = record.p_signal
        fs = record.fs

        limit = min(int(fs * 10), sig.shape[0])
        t = np.arange(limit) / fs

        # 如果导联数量变了再重建
        if len(self.axes) != num:
            self.fig.clear()
            self.axes = self.fig.subplots(num, 1, sharex=True)
            if num == 1:
                self.axes = [self.axes]

        for i, ax in enumerate(self.axes):
            ax.clear()
            ax.plot(t, sig[:limit, i], lw=0.8, color='#00FFCC')

            ax.set_facecolor('#121212')
            ax.set_yticks([])

            name = record.sig_name[i] if i < len(record.sig_name) else f"L{i+1}"
            ax.set_ylabel(
                name,
                rotation=0,
                labelpad=25,
                color='#AAAAAA',
                fontsize=10,
                va='center'
            )

            ax.grid(True, which='major', color='#333333', lw=0.6)
            ax.minorticks_on()
            ax.grid(True, which='minor', color='#222222', linestyle=':', lw=0.4)

            for s in ax.spines.values():
                s.set_color('#444444')

        self.axes[-1].set_xlabel("Time (s)", color='#AAAAAA')
        self.draw_idle()


class ECGAnnotator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.user = ""
        self.root = None

        self.tasks = []
        self.cur = -1
        self.recheck_mode = False

        self.db_path = Path(__file__).parent / "annotations.csv"
        self.df = self._load_db()

        self._init_ui()
        self._bind_keys()
        self._login()

    def _load_db(self):
        if self.db_path.exists():
            try:
                return pd.read_csv(
                    self.db_path,
                    encoding='utf-8-sig',
                    dtype={'filename': str}
                )
            except Exception:
                pass

        cols = [
            'filename',
            'foldername',
            'is_malignant',
            'annotator',
            'is_malignant_2nd',
            'annotator_2nd'
        ]
        return pd.DataFrame(columns=cols)

    def _init_ui(self):
        self.setWindowTitle("ECG Annotation Tool")
        self.showMaximized()

        self.setStyleSheet("""
            QMainWindow { background-color: #1A1A1A; }
            QWidget { color: #EEEEEE; font-family: 'Segoe UI', 'Microsoft YaHei UI'; }
            QPushButton {
                background-color: #2D2D2D;
                border: 1px solid #404040;
                padding: 12px;
                border-radius: 4px;
                min-width: 120px;
            }
            QPushButton:hover { background-color: #3D3D3D; }
            QPushButton:pressed { background-color: #505050; }
        """)

        main = QWidget()
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)

        side = QVBoxLayout()
        side.setContentsMargins(10, 10, 20, 10)

        self.status_label = QLabel("未加载数据")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "color:#FFCA28;font-size:14px;padding:15px;background:#262626;border-radius:6px;"
        )

        btn_load = QPushButton("加载数据")
        btn_load.clicked.connect(self.import_folder)

        btn_recheck = QPushButton("二次核验")
        btn_recheck.clicked.connect(self.start_recheck)

        btn_prev = QPushButton("← 上一个")
        btn_next = QPushButton("下一个 →")

        btn_prev.clicked.connect(self.prev)
        btn_next.clicked.connect(self.next)

        btn_yes = QPushButton("是 (1)")
        btn_no = QPushButton("否 (0)")
        btn_unknown = QPushButton("不确定 (999)")

        btn_yes.clicked.connect(lambda: self.save_and_next(1))
        btn_no.clicked.connect(lambda: self.save_and_next(0))
        btn_unknown.clicked.connect(lambda: self.save_and_next(999))

        side.addWidget(self.status_label)
        side.addSpacing(20)
        side.addWidget(btn_load)
        side.addWidget(btn_recheck)
        side.addSpacing(20)
        side.addWidget(btn_prev)
        side.addWidget(btn_next)
        side.addStretch()
        side.addWidget(btn_yes)
        side.addWidget(btn_no)
        side.addWidget(btn_unknown)

        self.canvas = ECGCanvas()

        layout.addLayout(side, 1)
        layout.addWidget(self.canvas, 6)

    def _bind_keys(self):
        mapping = {
            "1": lambda: self.save_and_next(1),
            "0": lambda: self.save_and_next(0),
            "9": lambda: self.save_and_next(999),
            "Left": self.prev,
            "Right": self.next
        }

        for k, func in mapping.items():
            shortcut = QKeySequence(k) if len(k) == 1 else getattr(Qt, f"Key_{k}")
            btn = QPushButton(self)
            btn.setShortcut(shortcut)
            btn.hide()
            btn.clicked.connect(func)

    def _login(self):
        name, ok = QInputDialog.getText(self, '标注ID', '请输入标注人员ID:')
        if not ok or not name.strip():
            sys.exit(0)

        self.user = name.strip()
        self.statusBar().showMessage(f"当前用户: {self.user}")

    def import_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择数据目录")
        if not path:
            return

        self.root = Path(path)
        self.recheck_mode = False

        done = set(self.df['filename'].astype(str).tolist())
        dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.tasks = [d for d in dirs if d.name not in done]

        if not self.tasks:
            QMessageBox.information(self, "完成", "全部文件已经标注完")
            return

        self.cur = 0
        self.show_current()

    def start_recheck(self):
        if self.df.empty:
            QMessageBox.warning(self, "提示", "没有历史记录")
            return

        mask = self.df['is_malignant'] == 999
        if 'is_malignant_2nd' in self.df.columns:
            mask = mask & self.df['is_malignant_2nd'].isna()

        targets = self.df[mask]['filename'].tolist()
        if not targets:
            QMessageBox.information(self, "完成", "没有需要二次复核的文件")
            return

        if not self.root:
            p = QFileDialog.getExistingDirectory(self, "重新选择数据目录")
            if not p:
                return
            self.root = Path(p)

        self.tasks = [
            self.root / str(f)
            for f in targets
            if (self.root / str(f)).exists()
        ]

        self.recheck_mode = True
        self.cur = 0
        self.show_current()

    def show_current(self):
        if self.cur < 0:
            return
        if self.cur >= len(self.tasks):
            return

        target = self.tasks[self.cur]
        prefix = str(target / target.name)

        try:
            record = wfdb.rdrecord(prefix)
            self.canvas.draw_record(record)

            tag = "[复核]" if self.recheck_mode else "[首轮]"
            self.status_label.setText(
                f"{tag}\n{target.name}\n({self.cur+1}/{len(self.tasks)})"
            )
            self.statusBar().showMessage("就绪")
        except Exception as e:
            self.status_label.setText(f"数据异常\n{target.name}")
            self.statusBar().showMessage(f"读取失败: {str(e)}")

            self.canvas.fig.clear()
            self.canvas.draw_idle()

    def save_and_next(self, val):
        if self.cur < 0:
            return
        if not self.tasks:
            return

        name = self.tasks[self.cur].name
        mask = self.df['filename'].astype(str) == name

        if mask.any():
            idx = self.df[mask].index[0]
            if self.recheck_mode:
                self.df.at[idx, 'is_malignant_2nd'] = val
                self.df.at[idx, 'annotator_2nd'] = self.user
            else:
                self.df.at[idx, 'is_malignant'] = val
                self.df.at[idx, 'annotator'] = self.user
        else:
            self.df.loc[len(self.df)] = {
                'filename': name,
                'foldername': self.root.name,
                'is_malignant': val,
                'annotator': self.user
            }

        # 每次都写盘，防止程序崩掉丢记录
        self.df.to_csv(self.db_path, index=False, encoding='utf-8-sig')

        self.next()

    def next(self):
        if self.cur >= len(self.tasks) - 1:
            self.statusBar().showMessage("全部完成")
            return

        self.cur += 1
        self.show_current()

    def prev(self):
        if self.cur <= 0:
            return

        self.cur -= 1
        self.show_current()


if __name__ == "__main__":
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    win = ECGAnnotator()
    win.show()

    sys.exit(app.exec_())