import numpy as np, pyqtgraph as pg, cv2
from PySide6.QtWidgets import QWidget, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage

from inference.SAM import set_image, segment_by_click
from inference.DINO import predict_image
from inference.CNN import predict_raman_file
from inference.fusion import fusion_predict


class ImageWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(760, 600); self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border:2px solid gray;background:#202020;")
        self.cb = None; self.ow = self.oh = self.dw = self.dh = None

    def show(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB); h,w,c = rgb.shape
        self.ow,self.oh = w,h
        q = QImage(rgb.data,w,h,c*w,QImage.Format_RGB888)
        p = QPixmap.fromImage(q).scaled(self.width(),self.height(),Qt.KeepAspectRatio)
        self.dw,self.dh = p.width(),p.height(); self.setPixmap(p)

    def load(self, path):
        img = cv2.imread(path)
        if img is None: raise ValueError(path)
        self.show(img)

    def mousePressEvent(self, e):
        if self.cb is None or self.ow is None: return
        x,y = e.position().x(),e.position().y()
        ox,oy = (self.width()-self.dw)/2,(self.height()-self.dh)/2
        if not (ox<=x<=ox+self.dw and oy<=y<=oy+self.dh): return
        self.cb(int((x-ox)*self.ow/self.dw), int((y-oy)*self.oh/self.dh))


class RamanWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(760,600)
        layout = QVBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        self.plot = pg.PlotWidget(); self.plot.setBackground("w")
        self.plot.setLabel("left","Normalized Intensity")
        self.plot.setLabel("bottom","Raman Shift",units="cm⁻¹")
        self.plot.getAxis("left").enableAutoSIPrefix(False)
        layout.addWidget(self.plot)

    def show(self,x,y):
        self.plot.clear()
        self.plot.plot(x,y,pen="r")
        self.plot.autoRange()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.image_label = self.image_prob = None
        self.raman_label = self.raman_prob = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Microplastic Multimodal Analysis")
        self.resize(1650,1050)

        self.image = ImageWidget(); self.image.cb = self.image_clicked
        self.raman = RamanWidget()

        b1,b2 = QPushButton("Upload Image"),QPushButton("Upload Spectrum")
        b1.clicked.connect(self.upload_image); b2.clicked.connect(self.upload_raman)

        left,right = QVBoxLayout(),QVBoxLayout()
        for layout,title,widget,button in [
            (left,"Image Analysis",self.image,b1),
            (right,"Raman Analysis",self.raman,b2)
        ]:
            t = QLabel(title); t.setStyleSheet("font-size:30px;font-weight:bold;")
            button.setFixedSize(760,45)
            layout.addWidget(t); layout.addWidget(widget); layout.addWidget(button)

        top = QHBoxLayout(); top.addLayout(left); top.addLayout(right)

        self.image_result = QLabel("Image Prediction : Waiting...")
        self.raman_result = QLabel("Raman Prediction : Waiting...")
        self.fusion_result = QLabel("Fusion Prediction : Waiting...")

        results = QVBoxLayout()
        title = QLabel("Prediction Results"); title.setStyleSheet("font-size:30px;font-weight:bold;")
        results.addWidget(title)

        style = "border:1px solid gray;background:#1e1e1e;font-size:24px;padding-left:20px;"
        for x in [self.image_result,self.raman_result,self.fusion_result]:
            x.setFixedSize(1550,70); x.setStyleSheet(style); results.addWidget(x)

        main = QVBoxLayout(self); main.addLayout(top); main.addLayout(results)

        self.setStyleSheet("""
        QWidget{background:#121212;color:white;}
        QPushButton{background:#2d2d2d;color:white;font-size:18px;}
        """)

    def upload_image(self):
        path,_ = QFileDialog.getOpenFileName(self,"Open Image","","Images (*.png *.jpg *.jpeg)")
        if not path: return
        self.image.load(path)
        set_image(cv2.imread(path))

    def image_clicked(self,x,y):
        overlay,crop,_ = segment_by_click(x,y)
        if overlay is None: return
        self.image.show(cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
        self.image_label,self.image_prob = predict_image(crop)
        self.image_result.setText(f"Image Prediction : {self.image_label} ")
        self.try_fusion()

    def upload_raman(self):
        path,_ = QFileDialog.getOpenFileName(self,"Open Spectrum","","Spectrum (*.txt)")
        if not path: return
        try:
            x,y,self.raman_label,self.raman_prob = predict_raman_file(path)
            self.raman.show(x,y)
            self.raman_result.setText(f"Raman Prediction : {self.raman_label} ")
            self.try_fusion()
        except Exception as e:
            print("Raman Error:",e)
            self.raman_label = None
            self.raman_result.setText("Raman Prediction : Error")

    def try_fusion(self):
        if self.image_label is None or self.raman_label is None: return
        result = fusion_predict(
            self.image_label,self.image_prob,
            self.raman_label,self.raman_prob
        )
        self.fusion_result.setText(f"Fusion Prediction : {result}")