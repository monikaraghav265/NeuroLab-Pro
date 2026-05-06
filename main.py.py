"""
NeuroLab Pro - Complete Neural Network Visualization Toolbox
Desktop Application using CustomTkinter
"""

import customtkinter as ctk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import threading
import time

# Import modules
from modules.perceptron import PerceptronLab
from modules.mlp import MLPLab
from modules.sentiment import SentimentLab
from modules.cnn import CNNLab
from modules.rnn import RNNLab
from modules.lstm import LSTMLab
from modules.hopfield import HopfieldLab

# Configure appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class NeuroLabApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("NeuroLab Pro - Neural Network Visualization Toolbox")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)
        
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Create sidebar
        self.create_sidebar()
        
        # Create main content area
        self.main_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # Current module reference
        self.current_module = None
        
        # Show dashboard by default
        self.show_dashboard()
        
    def create_sidebar(self):
        """Create sidebar with navigation buttons"""
        self.sidebar = ctk.CTkFrame(self.root, width=250, corner_radius=0, fg_color="#1a1a2e")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        
        # Logo/Title
        title_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        title_frame.pack(pady=20)
        
        title_label = ctk.CTkLabel(
            title_frame, 
            text="🧠 NeuroLab Pro", 
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color="#06b6d4"
        )
        title_label.pack()
        
        subtitle_label = ctk.CTkLabel(
            title_frame, 
            text="Complete AI Toolkit", 
            font=ctk.CTkFont(size=12),
            text_color="gray"
        )
        subtitle_label.pack()
        
        # Navigation buttons
        nav_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        nav_frame.pack(pady=20, fill="x", padx=15)
        
        modules = [
            ("📊 Dashboard", self.show_dashboard),
            ("🧠 Perceptron Lab", self.show_perceptron),
            ("🔗 MLP Lab", self.show_mlp),
            ("💬 Sentiment Analysis", self.show_sentiment),
            ("🎨 CNN Lab", self.show_cnn),
            ("🔄 RNN Lab", self.show_rnn),
            ("⏳ LSTM Lab", self.show_lstm),
            ("💾 Hopfield Lab", self.show_hopfield),
        ]
        
        self.nav_buttons = []
        for text, command in modules:
            btn = ctk.CTkButton(
                nav_frame,
                text=text,
                command=command,
                height=45,
                corner_radius=10,
                fg_color="transparent",
                text_color="white",
                hover_color="#2d2d5e",
                anchor="w"
            )
            btn.pack(fill="x", pady=5)
            self.nav_buttons.append(btn)
        
        # Footer
        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.pack(side="bottom", pady=20)
        
        status_label = ctk.CTkLabel(
            footer,
            text="🎓 8 Complete Modules\nReady for Training",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        status_label.pack()
        
    def clear_main_frame(self):
        """Clear the main content area"""
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
    def show_dashboard(self):
        """Display dashboard with analytics"""
        self.clear_main_frame()
        
        # Dashboard content
        dashboard_frame = ctk.CTkScrollableFrame(self.main_frame, fg_color="transparent")
        dashboard_frame.grid(row=0, column=0, sticky="nsew")
        
        # Title
        title = ctk.CTkLabel(
            dashboard_frame,
            text="Dashboard",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="#06b6d4"
        )
        title.pack(anchor="w", pady=(0, 20))
        
        # Stats cards
        stats_frame = ctk.CTkFrame(dashboard_frame, fg_color="transparent")
        stats_frame.pack(fill="x", pady=10)
        
        stats_frame.grid_columnconfigure((0,1,2,3), weight=1)
        
        stats = [
            ("🎯", "Average Accuracy", "96.8%", "#06b6d4"),
            ("📉", "Average Loss", "0.042", "#a855f7"),
            ("⚡", "Active Models", "8", "#10b981"),
            ("📚", "Labs Complete", "8/8", "#f59e0b")
        ]
        
        for i, (icon, title, value, color) in enumerate(stats):
            card = ctk.CTkFrame(stats_frame, corner_radius=15, fg_color="#1e1e3a")
            card.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            
            ctk.CTkLabel(card, text=icon, font=ctk.CTkFont(size=36)).pack(pady=(15, 5))
            ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12), text_color="gray").pack()
            ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=24, weight="bold"), text_color=color).pack(pady=(5, 15))
        
        # Quick launch section
        quick_frame = ctk.CTkFrame(dashboard_frame, corner_radius=15, fg_color="#1e1e3a")
        quick_frame.pack(fill="x", pady=20, padx=10)
        
        ctk.CTkLabel(quick_frame, text="🚀 Quick Launch Labs", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(15, 10))
        
        buttons_frame = ctk.CTkFrame(quick_frame, fg_color="transparent")
        buttons_frame.pack(fill="x", padx=20, pady=(0, 15))
        
        quick_launch = [
            ("🧠 Perceptron", self.show_perceptron),
            ("🔗 MLP", self.show_mlp),
            ("💬 Sentiment", self.show_sentiment),
            ("🎨 CNN", self.show_cnn),
            ("🔄 RNN", self.show_rnn),
            ("⏳ LSTM", self.show_lstm),
            ("💾 Hopfield", self.show_hopfield)
        ]
        
        for i, (text, cmd) in enumerate(quick_launch):
            btn = ctk.CTkButton(buttons_frame, text=text, command=cmd, height=40, corner_radius=10)
            btn.grid(row=i//4, column=i%4, padx=10, pady=10, sticky="ew")
            buttons_frame.grid_columnconfigure(i%4, weight=1)
        
        # Progress section
        progress_frame = ctk.CTkFrame(dashboard_frame, corner_radius=15, fg_color="#1e1e3a")
        progress_frame.pack(fill="x", pady=10, padx=10)
        
        ctk.CTkLabel(progress_frame, text="📈 Learning Progress", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(15, 10))
        
        progress_data = [
            ("Perceptron", 100, "#06b6d4"),
            ("MLP", 90, "#a855f7"),
            ("CNN", 75, "#10b981"),
            ("RNN", 60, "#f59e0b"),
            ("LSTM", 45, "#ef4444"),
            ("Hopfield", 30, "#8b5cf6")
        ]
        
        for name, value, color in progress_data:
            p_frame = ctk.CTkFrame(progress_frame, fg_color="transparent")
            p_frame.pack(fill="x", padx=20, pady=5)
            
            ctk.CTkLabel(p_frame, text=name, font=ctk.CTkFont(size=12)).pack(side="left")
            ctk.CTkLabel(p_frame, text=f"{value}%", font=ctk.CTkFont(size=12), text_color=color).pack(side="right")
            
            progress_bar = ctk.CTkProgressBar(p_frame, height=8, corner_radius=4, progress_color=color)
            progress_bar.pack(fill="x", pady=5)
            progress_bar.set(value/100)
        
        # Recent activity
        activity_frame = ctk.CTkFrame(dashboard_frame, corner_radius=15, fg_color="#1e1e3a")
        activity_frame.pack(fill="x", pady=20, padx=10)
        
        ctk.CTkLabel(activity_frame, text="🎯 Recent Activity", font=ctk.CTkFont(size=18, weight="bold")).pack(anchor="w", padx=20, pady=(15, 10))
        
        activities = [
            ("✅", "Perceptron trained on AND gate - 100% accuracy", "Just now"),
            ("🔄", "MLP backpropagation visualization ready", "2 mins ago"),
            ("📊", "Dashboard analytics updated", "5 mins ago")
        ]
        
        for icon, msg, time_str in activities:
            a_frame = ctk.CTkFrame(activity_frame, fg_color="#2d2d5e", corner_radius=10)
            a_frame.pack(fill="x", padx=20, pady=5)
            
            ctk.CTkLabel(a_frame, text=icon, font=ctk.CTkFont(size=20)).pack(side="left", padx=10)
            
            text_frame = ctk.CTkFrame(a_frame, fg_color="transparent")
            text_frame.pack(side="left", fill="x", expand=True, padx=10, pady=10)
            
            ctk.CTkLabel(text_frame, text=msg, font=ctk.CTkFont(size=12), anchor="w").pack(anchor="w")
            ctk.CTkLabel(text_frame, text=time_str, font=ctk.CTkFont(size=10), text_color="gray").pack(anchor="w")
    
    def show_perceptron(self):
        """Show Perceptron Lab"""
        self.clear_main_frame()
        perceptron_lab = PerceptronLab(self.main_frame)
        perceptron_lab.pack(fill="both", expand=True)
    
    def show_mlp(self):
        """Show MLP Lab"""
        self.clear_main_frame()
        mlp_lab = MLPLab(self.main_frame)
        mlp_lab.pack(fill="both", expand=True)
    
    def show_sentiment(self):
        """Show Sentiment Analysis Lab"""
        self.clear_main_frame()
        sentiment_lab = SentimentLab(self.main_frame)
        sentiment_lab.pack(fill="both", expand=True)
    
    def show_cnn(self):
        """Show CNN Lab"""
        self.clear_main_frame()
        cnn_lab = CNNLab(self.main_frame)
        cnn_lab.pack(fill="both", expand=True)
    
    def show_rnn(self):
        """Show RNN Lab"""
        self.clear_main_frame()
        rnn_lab = RNNLab(self.main_frame)
        rnn_lab.pack(fill="both", expand=True)
    
    def show_lstm(self):
        """Show LSTM Lab"""
        self.clear_main_frame()
        lstm_lab = LSTMLab(self.main_frame)
        lstm_lab.pack(fill="both", expand=True)
    
    def show_hopfield(self):
        """Show Hopfield Lab"""
        self.clear_main_frame()
        hopfield_lab = HopfieldLab(self.main_frame)
        hopfield_lab.pack(fill="both", expand=True)
    
    def run(self):
        """Run the application"""
        self.root.mainloop()


if __name__ == "__main__":
    app = NeuroLabApp()
    app.run()