import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime
import os
import cv2
import shutil
import models_imp

class CameraCaptureApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.title("Camera Capture")
        self.geometry("800x600")
        
        self.capture_button = tk.Button(self, text="Start Capture", command=self.toggle_capture)
        self.capture_button.pack(pady=10)
        
        self.train_button = tk.Button(self, text="Train", command=self.train_model)
        self.train_button.pack(pady=10)
        
        self.predict_button = tk.Button(self, text="Predict", command=self.toggle_prediction)
        self.predict_button.pack(pady=10)

        self.predict_label = tk.Label(self, text="False")
        self.predict_label.pack(pady=10)
        
        self.video_feed_label = tk.Label(self)
        self.video_feed_label.pack()
        
        self.capturing = False
        self.predicting = False
        self.frame_counter = 0  # Counter to keep track of frames
        self.initialize_camera()
        self.update_video_feed()
        
        self.current_class = 0
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def initialize_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
    def release_camera(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        
    def toggle_capture(self):
        if not self.capturing:
            self.capturing = True
            self.capture_button.config(text="Stop Capture")
            self.start_time = datetime.now()
            self.create_class_folder(self.current_class)
        else:
            self.capturing = False
            self.capture_button.config(text="Start Capture")
            self.current_class += 1
    
    def create_class_folder(self, class_idx):
        save_folder = os.path.join("captured_frames", f"class{class_idx}")
        
        if os.path.exists(save_folder):
            response = messagebox.askyesnocancel("Folder Exists", f"Folder for class {class_idx} already exists. Do you want to overwrite it?")
            if response is True:  # Yes, overwrite
                shutil.rmtree(save_folder)
                os.makedirs(save_folder)
            elif response is False:  # No, append
                return
            else:  # Cancel
                self.capturing = False
                self.capture_button.config(text="Start Capture")
                return
        else:
            os.makedirs(save_folder)
        
    def save_frame(self, frame, class_idx):
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
        save_folder = os.path.join("captured_frames", f"class{class_idx}")
        cv2.imwrite(os.path.join(save_folder, filename), frame)
        
    def train_model(self):
        print("Training the model...")
        self.train_button.config(text="Training!!!")
        
        dataset, num_classes = models_imp.generate_dataset(os.path.join("captured_frames"))
        img, lab = models_imp.load_dataset(dataset)
        print(img.shape)
        train_x, train_y = models_imp.train_set(img, lab, num_classes)
        print(train_x.shape)
        model_path = os.path.join("model.keras")
        models_imp.train_model(train_x, train_y,model_path, num_classes)
        print("Model trained successfully.")
        self.train_button.config(text="Train")
        
    def toggle_prediction(self):
        self.predicting = not self.predicting
        if not self.predicting:
            self.predict_label.config(text="False")
        
    def predict_class(self, frame):
        max = 0
        label = 0
        processed_frame = models_imp.preprocess_images([frame])
        feature = models_imp.test_set(processed_frame)
        print(feature.shape)
        model_path = os.path.join("model.keras")
        predictions = models_imp.predict(model_path, feature)
        print(predictions)
        for j in range(predictions.shape[0]):
            for i in range(predictions.shape[1]):
                if(predictions[j][i] > max):
                    label = i
                    max = predictions[j][i]
        return label
        
    def update_video_feed(self):
        success, frame = self.cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.video_feed_label.img = img
            self.video_feed_label.config(image=img)
            
            if self.capturing and (datetime.now() - self.start_time).total_seconds() < 50:
                self.save_frame(frame, self.current_class)
            
            if self.predicting:
                self.frame_counter += 1
                if self.frame_counter % 15 == 0:
                    predictions = self.predict_class(frame)
                    
                    self.predict_label.config(text=f"Prediction: {predictions}")
                    #print(predictions)

        self.after(10, self.update_video_feed)

    def on_closing(self):
        self.release_camera()
        self.destroy()

if __name__ == "__main__":
    app = CameraCaptureApp()
    app.mainloop()
