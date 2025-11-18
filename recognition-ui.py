import customtkinter as ctk
import cv2
from PIL import Image
from deepface import DeepFace
import threading

# --- Configuration ---
ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class EmotionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. Setup the Main Window
        self.title("AI Emotion Detector")
        self.geometry("900x600")

        # 2. Create the Layout Grid
        self.grid_columnconfigure(0, weight=1) # Left side (Video)
        self.grid_columnconfigure(1, weight=0) # Right side (Controls)
        self.grid_rowconfigure(0, weight=1)

        # --- Left Side: Video Feed ---
        self.video_frame = ctk.CTkFrame(self, corner_radius=15)
        self.video_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # --- Right Side: Dashboard ---
        self.control_frame = ctk.CTkFrame(self, width=250, corner_radius=15)
        self.control_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        # Title Label
        self.title_label = ctk.CTkLabel(self.control_frame, text="Live Analysis", 
                                      font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(30, 10))

        # The Emotion Display (Big Text)
        self.emotion_label = ctk.CTkLabel(self.control_frame, text="Neutral", 
                                        font=ctk.CTkFont(size=30, weight="bold"),
                                        text_color="#3B8ED0") # Modern Blue
        self.emotion_label.pack(pady=20)

        # Quit Button
        self.btn_quit = ctk.CTkButton(self.control_frame, text="STOP CAMERA", 
                                    fg_color="#D9534F", hover_color="#C9302C", # Red color
                                    command=self.close_app)
        self.btn_quit.pack(side="bottom", pady=30)

        # --- Backend Logic ---
        self.cap = cv2.VideoCapture(0)
        self.current_emotion = "Neutral"
        self.running = True
        
        # Start the AI thread (Background Brain)
        self.thread = threading.Thread(target=self.analyze_emotion, daemon=True)
        self.thread.start()

        # Start the Video Loop (Frontend Eyes)
        self.update_video()

    def analyze_emotion(self):
        """Runs in background to keep UI smooth"""
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                frame_count += 1
                # Analyze every 30 frames to save resources
                if frame_count % 30 == 0:
                    try:
                        # enforce_detection=False prevents crash if no face detected
                        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                        if isinstance(result, list): result = result[0]
                        
                        emotion = result['dominant_emotion']
                        self.current_emotion = emotion.capitalize()
                        
                        # Update the big text label color based on emotion
                        self.update_emotion_color(emotion)
                        
                    except Exception as e:
                        print(e)

    def update_emotion_color(self, emotion):
        """Changes the text color dynamically"""
        colors = {
            "angry": "#D9534F",  # Red
            "happy": "#5CB85C",  # Green
            "sad": "#5BC0DE",    # Light Blue
            "neutral": "#F7F7F7" # White/Grey
        }
        # .after() ensures we update GUI from the main thread safely
        color = colors.get(emotion, "#F7F7F7")
        self.emotion_label.configure(text=self.current_emotion, text_color=color)

    def update_video(self):
        """Reads video frame and converts to CustomTkinter format"""
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                # 1. Convert Color (OpenCV is BGR, GUI needs RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 2. Convert to PIL Image
                img = Image.fromarray(frame_rgb)
                
                # 3. Create CTkImage (handles high DPI screens better)
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 480))
                
                # 4. Update Label
                self.video_label.configure(image=ctk_img)
                self.video_label.image = ctk_img # Keep reference
            
            # Call this function again in 10ms
            self.after(10, self.update_video)

    def close_app(self):
        self.running = False
        self.cap.release()
        self.destroy()

if __name__ == "__main__":
    app = EmotionApp()
    app.mainloop()