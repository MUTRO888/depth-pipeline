import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import yaml
from PIL import Image, ImageTk
import threading

# Optional drag-and-drop support
try:
    import sys
    if sys.platform == 'darwin':
        raise ImportError("macOS tkinterdnd2 binaries are often broken, falling back to standard Tk")
    from tkinterdnd2 import TkinterDnD, DND_FILES

    HAS_DND = True
except ImportError:
    HAS_DND = False

# Ensure sibling packages are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import DepthPipeline, get_config_path
from utils.image_io import save_bmp

THUMBNAIL_MAX = 250
PREVIEW_MAX = 300


class DepthPipelineApp:
    def __init__(self, root):
        self.root = root
        self.root.title("浮雕深度估计")
        self.root.geometry("860x780")
        self.root.resizable(True, True)

        self.image_path = None
        self.original_image = None
        self.result_image = None
        self.pipeline = None
        self.processing = False

        # prevent garbage-collection of PhotoImage objects
        self._tk_images = []

        self._build_ui()

    # ── UI Construction ──────────────────────────────────────────────

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        self._build_upload_section(main)
        ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        self._build_settings_section(main)
        ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        self._build_status_section(main)
        ttk.Separator(main, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        self._build_export_section(main)

    def _build_upload_section(self, parent):
        frame = ttk.LabelFrame(parent, text="上传区域", padding=10)
        frame.pack(fill=tk.X)

        # Drop / click zone
        self.drop_zone = tk.Frame(
            frame, bg="#f0f0f0", height=200, cursor="hand2", relief=tk.GROOVE, bd=2
        )
        self.drop_zone.pack(fill=tk.X, pady=5)
        self.drop_zone.pack_propagate(False)

        self.drop_label = tk.Label(
            self.drop_zone,
            text="点击此处选择图片，或将图片拖拽到这里\n支持 JPG / PNG 格式",
            bg="#f0f0f0",
            fg="#666666",
            font=("", 12),
        )
        self.drop_label.pack(expand=True)

        self.drop_zone.bind("<Button-1>", lambda e: self._open_file_dialog())
        self.drop_label.bind("<Button-1>", lambda e: self._open_file_dialog())

        if HAS_DND:
            self.drop_zone.drop_target_register(DND_FILES)
            self.drop_zone.dnd_bind("<<Drop>>", self._on_drop)

        # Info row + Start button
        bottom_frame = ttk.Frame(frame)
        bottom_frame.pack(fill=tk.X, pady=5)
        
        self.file_info_label = ttk.Label(bottom_frame, text="")
        self.file_info_label.pack(side=tk.LEFT)

        self.start_btn = ttk.Button(
            bottom_frame, text="开始处理 (三路融合)", command=self._start_processing, state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.RIGHT)

    def _build_settings_section(self, parent):
        frame = ttk.LabelFrame(parent, text="浮雕细节设置", padding=10)
        frame.pack(fill=tk.X)
        
        # Load initial values from config
        config_path = get_config_path()
        try:
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                fw = cfg.get("fusion", {})
                ds = fw.get("detail_strength", 1.5)
                rs = fw.get("relief_strength", 0.5)
        except Exception:
            ds, rs = 1.5, 0.5
            
        self.var_detail = tk.DoubleVar(value=ds)
        self.var_relief = tk.DoubleVar(value=rs)
        
        def on_slider_change(*args):
            self.lbl_detail_val.config(text=f"{self.var_detail.get():.1f}x")
            self.lbl_relief_val.config(text=f"{self.var_relief.get():.1f}x")
                
        # Detail strength
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="原图细节注入强度:", width=20).pack(side=tk.LEFT)
        s1 = ttk.Scale(row1, from_=0.0, to=3.0, variable=self.var_detail, orient=tk.HORIZONTAL, command=on_slider_change)
        s1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.lbl_detail_val = ttk.Label(row1, text="-", width=6)
        self.lbl_detail_val.pack(side=tk.LEFT)
        
        # Relief strength (SD LoRA)
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="SD微纹理注入强度:", width=20).pack(side=tk.LEFT)
        s2 = ttk.Scale(row2, from_=0.0, to=3.0, variable=self.var_relief, orient=tk.HORIZONTAL, command=on_slider_change)
        s2.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.lbl_relief_val = ttk.Label(row2, text="-", width=6)
        self.lbl_relief_val.pack(side=tk.LEFT)
        
        on_slider_change()

    def _build_status_section(self, parent):
        frame = ttk.LabelFrame(parent, text="处理状态", padding=10)
        frame.pack(fill=tk.X)

        self.status_label = ttk.Label(frame, text="", font=("", 11))
        self.status_label.pack(pady=5)

        self.progress = ttk.Progressbar(frame, mode="indeterminate", length=400)
        # hidden until depth estimation starts

    def _build_export_section(self, parent):
        self.export_frame = ttk.LabelFrame(parent, text="导出区域", padding=10)
        self.export_frame.pack(fill=tk.BOTH, expand=True)

        # ── placeholder (visible before processing completes) ──
        self.export_placeholder = ttk.Label(
            self.export_frame,
            text="上传图片并处理后，预览将显示在此处",
            foreground="#999999",
            font=("", 11),
        )
        self.export_placeholder.pack(expand=True, pady=30)

        # ── preview row (hidden until processing completes) ──
        self.preview_frame = ttk.Frame(self.export_frame)

        left = ttk.Frame(self.preview_frame)
        left.pack(side=tk.LEFT, expand=True, padx=5)
        ttk.Label(left, text="原图").pack()
        self.orig_preview_label = ttk.Label(left)
        self.orig_preview_label.pack()

        right = ttk.Frame(self.preview_frame)
        right.pack(side=tk.LEFT, expand=True, padx=5)
        ttk.Label(right, text="灰度深度图").pack()
        self.depth_preview_label = ttk.Label(right)
        self.depth_preview_label.pack()

        # ── path row (always visible) ──
        self.path_frame = ttk.Frame(self.export_frame)
        ttk.Label(self.path_frame, text="导出路径:").pack(side=tk.LEFT)
        self.path_var = tk.StringVar()
        self.path_entry = ttk.Entry(self.path_frame, textvariable=self.path_var)
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(self.path_frame, text="浏览", command=self._browse_save_path).pack(
            side=tk.LEFT
        )
        self.path_frame.pack(fill=tk.X, pady=5)

        # ── save button (always visible, disabled until result ready) ──
        self.save_btn = ttk.Button(
            self.export_frame,
            text="保存文件",
            command=self._save_file,
            state=tk.DISABLED,
        )
        self.save_btn.pack(pady=5)

    # ── Upload Handling ──────────────────────────────────────────────

    def _open_file_dialog(self):
        if self.processing:
            return
        path = filedialog.askopenfilename(
            filetypes=[("图片文件", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if path:
            self._load_image(path)

    def _on_drop(self, event):
        if self.processing:
            return
        path = event.data
        # Windows: paths with spaces are wrapped in braces
        if path.startswith("{") and path.endswith("}"):
            path = path[1:-1]
        self._load_image(path.strip())

    def _load_image(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            messagebox.showerror("错误", "请上传 JPG 或 PNG 格式的图片")
            return

        try:
            img = Image.open(path)
            img.load()
        except Exception:
            messagebox.showerror("错误", "文件读取失败，请重新选择")
            return

        self.image_path = path
        self.original_image = img
        self.result_image = None

        self._hide_export()
        self.status_label.config(text="")
        self._show_thumbnail(img)

        w, h = img.size
        name = os.path.basename(path)
        self.file_info_label.config(text=f"{name}  |  {w} × {h} px")
        self.start_btn.config(state=tk.NORMAL)

    def _show_thumbnail(self, img):
        for w in self.drop_zone.winfo_children():
            w.destroy()

        thumb = img.copy()
        thumb.thumbnail((THUMBNAIL_MAX, THUMBNAIL_MAX))
        tk_img = ImageTk.PhotoImage(thumb)
        self._tk_images = [tk_img]

        label = tk.Label(self.drop_zone, image=tk_img, bg="#f0f0f0", cursor="hand2")
        label.pack(expand=True)
        label.bind("<Button-1>", lambda e: self._open_file_dialog())

    # ── Processing ───────────────────────────────────────────────────

    def _start_processing(self):
        if self.processing or not self.image_path:
            return

        self.processing = True
        self.start_btn.config(state=tk.DISABLED)
        self._hide_export()
        self.status_label.config(text="")

        thread = threading.Thread(target=self._process_worker, daemon=True)
        thread.start()

    def _process_worker(self):
        try:
            config_path = get_config_path()
            
            # Save current slider settings to config memory before run 
            # Note: We don't overwrite the original YAML back to disk to save lifetime,
            # but we pass the new weights into the internal config object.
            
            def status_cb(msg):
                self.root.after(0, lambda m=msg: self._update_status(m))

            if self.pipeline is None:
                self.pipeline = DepthPipeline(config_path)
                
            # Update GUI values into pipeline config before processing
            if "fusion" not in self.pipeline.config:
                self.pipeline.config["fusion"] = {}
            self.pipeline.config["fusion"]["detail_strength"] = self.var_detail.get()
            self.pipeline.config["fusion"]["relief_strength"] = self.var_relief.get()

            original, result = self.pipeline.process(
                self.image_path, status_callback=status_cb
            )

            self.root.after(0, lambda: self._on_complete(original, result))

        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: self._on_error(error_msg))

    def _update_status(self, msg):
        self.status_label.config(text=msg)
        if "分析" in msg or "渲染" in msg or "融合" in msg:
            self._show_progress()
        elif "显存不足" in msg or "首次运行" in msg or "正在下载" in msg:
            self._show_progress()
        else:
            self._hide_progress()

    def _show_progress(self):
        self.progress.pack(pady=5)
        self.progress.start(15)

    def _hide_progress(self):
        self.progress.stop()
        self.progress.pack_forget()

    def _on_complete(self, original, result):
        self.processing = False
        self.result_image = result
        self.start_btn.config(state=tk.NORMAL)

        self._show_export(original, result)

        base, _ = os.path.splitext(self.image_path)
        self.path_var.set(f"{base}_depth.bmp")

    def _on_error(self, error_msg):
        self.processing = False
        self._hide_progress()
        self.status_label.config(text=f"处理失败，请重试\n{error_msg}")
        self.start_btn.config(state=tk.NORMAL)

    # ── Export ───────────────────────────────────────────────────────

    def _show_export(self, original, result):
        self.export_placeholder.pack_forget()

        orig_thumb = original.copy()
        orig_thumb.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
        tk_orig = ImageTk.PhotoImage(orig_thumb)

        result_thumb = result.copy()
        result_thumb.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
        tk_result = ImageTk.PhotoImage(result_thumb)

        self._tk_images = [tk_orig, tk_result]

        self.orig_preview_label.config(image=tk_orig)
        self.depth_preview_label.config(image=tk_result)

        # Insert preview above the path row
        self.preview_frame.pack(fill=tk.BOTH, expand=True, pady=5, before=self.path_frame)
        self.save_btn.config(state=tk.NORMAL, text="保存文件")

    def _hide_export(self):
        self.preview_frame.pack_forget()
        self.save_btn.config(state=tk.DISABLED, text="保存文件")
        self.path_var.set("")
        self.export_placeholder.pack(expand=True, pady=30, before=self.path_frame)

    def _browse_save_path(self):
        current = self.path_var.get()
        path = filedialog.asksaveasfilename(
            defaultextension=".bmp",
            filetypes=[("BMP 文件", "*.bmp")],
            initialfile=os.path.basename(current) if current else "output_depth.bmp",
        )
        if path:
            self.path_var.set(path)

    def _save_file(self):
        if self.result_image is None:
            return

        path = self.path_var.get()
        if not path:
            messagebox.showerror("错误", "请指定导出路径")
            return

        if os.path.exists(path):
            if not messagebox.askyesno("确认", "文件已存在，是否覆盖？"):
                return

        try:
            save_bmp(self.result_image, path)
            self.save_btn.config(text="已保存 ✓", state=tk.DISABLED)
            self.root.after(
                2000,
                lambda: self.save_btn.config(text="保存文件", state=tk.NORMAL),
            )
        except PermissionError:
            messagebox.showerror("错误", "无法写入该目录，请选择其他位置")
        except OSError as e:
            if "space" in str(e).lower() or "disk" in str(e).lower():
                messagebox.showerror("错误", "磁盘空间不足")
            else:
                messagebox.showerror("错误", f"保存失败: {e}")


# ── Entry point ──────────────────────────────────────────────────────

def main():
    print(">> 正在初始化图形界面 (Tkinter)...")
    if HAS_DND:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
        
    # 如果是 Mac，让窗口强制置顶一次避免藏在终端后面
    if sys.platform == 'darwin':
        os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
        
    app = DepthPipelineApp(root)
    print(">> 启动成功！如果没看到窗口，请在【程序坞（底部任务栏）】里找一下“Python”或者羽毛笔图标。")
    root.mainloop()

if __name__ == "__main__":
    main()
