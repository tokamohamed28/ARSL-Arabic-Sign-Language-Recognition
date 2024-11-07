import cv2
import torch
import supervision as sv
from ultralytics import YOLOv10
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from bidi.algorithm import get_display
import arabic_reshaper

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextBrowser, QDialog
from PyQt5.QtGui import QImage, QTextCursor
import imutils

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("ARSL")
        Dialog.resize(1121, 853)
        
        # Layouts
        self.main_layout = QVBoxLayout(Dialog)
        self.button_layout = QVBoxLayout()
        self.display_layout = QHBoxLayout()

        # Buttons
        self.pushButton = QPushButton(Dialog)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QPushButton(Dialog)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QPushButton(Dialog)
        self.pushButton_3.setObjectName("pushButton_3")

        # Add buttons to layout
        self.button_layout.addWidget(self.pushButton)
        self.button_layout.addWidget(self.pushButton_2)
        self.button_layout.addWidget(self.pushButton_3)

        # Label for video
        self.label = QLabel(Dialog)
        self.label.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label.setAutoFillBackground(False)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("ambulance_icon.png"))
        self.label.setObjectName("label")

        # Text Browser
        self.textBrowser = QTextBrowser(Dialog)
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setStyleSheet('font-size: 30px;')

        # Add widgets to display layout
        self.display_layout.addWidget(self.label)
        self.display_layout.addLayout(self.button_layout)

        # Add display layout and text browser to main layout
        self.main_layout.addLayout(self.display_layout)
        self.main_layout.addWidget(self.textBrowser)

        self.retranslateUi(Dialog)
        self.pushButton.clicked.connect(self.toggle_camera)  # type: ignore
        self.pushButton_2.clicked.connect(self.clear_global_word)  # type: ignore
        self.pushButton_3.clicked.connect(self.add_space)  # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        self.filename = None
        self.temp = None
        self.started = False
        self.global_word = ""

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        self.last_label = None
        self.cap = None
        self.model = None
        self.bounding_box_annotator = None
        self.font = None

        # Set to track displayed labels
        self.displayed_labels = set()

    def clear_global_word(self):
        self.textBrowser.clear()
        self.global_word = ""
        self.displayed_labels.clear()  # Clear the set when the textBrowser is cleared

    def toggle_camera(self):
        if self.started:
            self.started = False
            self.pushButton.setText('Start')
            self.timer.stop()
            if self.cap:
                self.cap.release()
        else:
            self.started = True
            self.pushButton.setText('Stop')
            self.init_camera()
            self.timer.start(30)  # Update every 30 ms

    def init_camera(self):
        # Ensure GPU availability and compatibility
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Mapping for Arabic sign language gestures
        mewnames = {
            0: 'ع', 1: 'ال', 2: 'ج', 3:'ك', 4: 'خ', 
            5: 'لا', 6: 'ل', 7: 'م', 8: 'ن', 9: 'ق', 
            10:'ر', 11: 'ص', 12: 'أ', 13: 'س', 14: 'ش', 
            15: 'ط', 16: 'ت', 17: 'ة', 18: 'ذ', 19: 'ث', 
            20: 'و', 21: 'ي', 22: 'ظ', 23: 'ب', 24: 'ز', 
            25: 'ض', 26: 'د', 27: 'ف', 28: 'غ', 29: 'ح', 
            30: 'ه', 31: 'أنا', 32: 'أحبك', 33: 'عمري', 34: 'اسمي'
        }
        
        # Load YOLO model with specified names
        self.model = YOLOv10('bestnan.pt', names=mewnames)
        self.bounding_box_annotator = sv.BoundingBoxAnnotator()
        self.cap = cv2.VideoCapture(0)
        font_path = "NotoSansArabic-Light.ttf"  # Path to your TTF font file
        self.font = ImageFont.truetype(font_path, 20)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        results = self.model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        annotated_image = self.bounding_box_annotator.annotate(scene=frame, detections=detections)

        # Convert frame to PIL image
        annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(annotated_image_pil)

        # Annotate labels in Arabic and update textBrowser
        labels_detected = set()
        for detection in detections:
            bbox = detection[0]  # Bounding box coordinates
            label_dict = detection[5]  # Assuming label dictionary is at index 5

            if isinstance(label_dict, dict):
                label = list(label_dict.values())[0]  # Retrieve the first value from the dictionary

                # Reshape the Arabic label
                label = arabic_reshaper.reshape(label)

                # Apply bidirectional text rendering to the label
                label = get_display(label)

                # Draw text with PIL
                draw.text((bbox[0], bbox[1]), label, font=self.font, fill=(255, 0, 0))

                # Add to the set of detected labels
                labels_detected.add(label)

        # Convert back to OpenCV image
        annotated_image = cv2.cvtColor(np.array(annotated_image_pil), cv2.COLOR_RGB2BGR)
        self.setPhoto(annotated_image)

        # Update the textBrowser with detected labels
        self.update_text_browser(labels_detected)

    def update_text_browser(self, labels_detected):
        for label in labels_detected:
            # Skip adding the label if it is the same as the last added label
            if label == self.last_label:
                continue
            
            reshaped_label = arabic_reshaper.reshape(label)
            displayed_label = get_display(reshaped_label)
            self.global_word += "" + displayed_label
            self.last_label = label  # Update last_label to the current label
            
        self.textBrowser.setPlainText(self.global_word.strip())
        self.displayed_labels.update(labels_detected)


    def add_space(self):
        self.global_word += " "
        reshaped_global_word = arabic_reshaper.reshape(self.global_word)
        displayed_global_word = get_display(reshaped_global_word)
        self.textBrowser.setPlainText(displayed_global_word)

    def setPhoto(self, image):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        self.temp = image
        image = imutils.resize(image, width=1121, height=853)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "ARSL"))
        self.pushButton.setText(_translate("Dialog", "Start"))
        self.pushButton_2.setText(_translate("Dialog", "Clear"))
        self.pushButton_3.setText(_translate("Dialog", "Add Space"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
