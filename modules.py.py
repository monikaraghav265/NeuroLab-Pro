"""
Perceptron Lab Module - Single Layer Neural Network
"""

import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading


class Perceptron:
    """Perceptron neural network implementation"""
    
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.randn(input_size) * 0.5
        self.bias = np.random.randn() * 0.5
        self.learning_rate = learning_rate
        self.history = {'loss': [], 'accuracy': []}
    
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activation(z)
    
    def train(self, X, y, epochs, callback=None):
        self.history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                total_loss += error ** 2
                
                # Update weights
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
                
                if prediction == y[i]:
                    correct += 1
            
            avg_loss = total_loss / len(X)
            accuracy = (correct / len(X)) * 100
            
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(accuracy)
            
            if callback:
                callback(epoch, avg_loss, accuracy)
        
        return {
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'final_accuracy': self.history['accuracy'][-1],
            'final_loss': self.history['loss'][-1]
        }


class PerceptronLab(ctk.CTkFrame):
    """Perceptron Lab GUI Component"""
    
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        
        self.perceptron = None
        self.training_active = False
        
        self.setup_ui()
        self.datasets = {
            'AND': {'X': np.array([[0,0], [0,1], [1,0], [1,1]]), 'y': np.array([0,0,0,1])},
            'OR': {'X': np.array([[0,0], [0,1], [1,0], [1,1]]), 'y': np.array([0,1,1,1])},
            'NAND': {'X': np.array([[0,0], [0,1], [1,0], [1,1]]), 'y': np.array([1,1,1,0])},
            'XOR': {'X': np.array([[0,0], [0,1], [1,0], [1,1]]), 'y': np.array([0,1,1,0])}
        }
    
    def setup_ui(self):
        """Setup the user interface"""
        # Title
        title = ctk.CTkLabel(
            self, 
            text="🧠 Perceptron Lab - Single Layer Neural Network",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color="#06b6d4"
        )
        title.pack(anchor="w", padx=20, pady=(20, 10))
        
        subtitle = ctk.CTkLabel(
            self,
            text="Learn the fundamentals of binary classification with interactive training",
            font=ctk.CTkFont(size=14),
            text_color="gray"
        )
        subtitle.pack(anchor="w", padx=20, pady=(0, 20))
        
        # Main content - split into two columns
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)
        content_frame.grid_columnconfigure(0, weight=1)
        content_frame.grid_columnconfigure(1, weight=2)
        
        # Left Panel - Controls
        self.create_controls_panel(content_frame)
        
        # Right Panel - Visualization
        self.create_visualization_panel(content_frame)
    
    def create_controls_panel(self, parent):
        """Create left panel with training controls"""
        left_panel = ctk.CTkFrame(parent, corner_radius=15, fg_color="#1e1e3a")
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=10)
        
        # Controls title
        ctk.CTkLabel(
            left_panel, 
            text="🎮 Training Controls",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(anchor="w", padx=20, pady=(20, 15))
        
        # Dataset selection
        ctk.CTkLabel(left_panel, text="Dataset", font=ctk.CTkFont(size=14)).pack(anchor="w", padx=20, pady=(10, 5))
        self.dataset_var = ctk.StringVar(value="AND")
        dataset_menu = ctk.CTkOptionMenu(
            left_panel,
            values=["AND", "OR", "NAND", "XOR"],
            variable=self.dataset_var,
            width=200
        )
        dataset_menu.pack(anchor="w", padx=20, pady=(0, 15))
        
        # Learning rate slider
        ctk.CTkLabel(left_panel, text="Learning Rate", font=ctk.CTkFont(size=14)).pack(anchor="w", padx=20, pady=(10, 5))
        self.lr_var = ctk.DoubleVar(value=0.1)
        lr_slider = ctk.CTkSlider(left_panel, from_=0.01, to=1.0, variable=self.lr_var, number_of_steps=99)
        lr_slider.pack(fill="x", padx=20, pady=(0, 5))
        self.lr_label = ctk.CTkLabel(left_panel, text="0.10", font=ctk.CTkFont(size=12), text_color="gray")
        self.lr_label.pack(anchor="w", padx=20, pady=(0, 15))
        
        # Epochs slider
        ctk.CTkLabel(left_panel, text="Epochs", font=ctk.CTkFont(size=14)).pack(anchor="w", padx=20, pady=(10, 5))
        self.epochs_var = ctk.IntVar(value=50)
        epochs_slider = ctk.CTkSlider(left_panel, from_=10, to=200, variable=self.epochs_var, number_of_steps=19)
        epochs_slider.pack(fill="x", padx=20, pady=(0, 5))
        self.epochs_label = ctk.CTkLabel(left_panel, text="50", font=ctk.CTkFont(size=12), text_color="gray")
        self.epochs_label.pack(anchor="w", padx=20, pady=(0, 15))
        
        # Train button
        self.train_btn = ctk.CTkButton(
            left_panel,
            text="🚀 Start Training",
            command=self.start_training,
            height=45,
            corner_radius=10,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.train_btn.pack(fill="x", padx=20, pady=(20, 10))
        
        # Reset button
        reset_btn = ctk.CTkButton(
            left_panel,
            text="🔄 Reset",
            command=self.reset,
            height=40,
            corner_radius=10,
            fg_color="transparent",
            border_width=1,
            border_color="#a855f7"
        )
        reset_btn.pack(fill="x", padx=20, pady=(0, 20))
        
        # Stats panel
        stats_panel = ctk.CTkFrame(left_panel, corner_radius=10, fg_color="#2d2d5e")
        stats_panel.pack(fill="x", padx=20, pady=20)
        
        ctk.CTkLabel(
            stats_panel, 
            text="📊 Training Stats",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(anchor="w", padx=15, pady=(15, 10))
        
        # Status
        status_frame = ctk.CTkFrame(stats_panel, fg_color="transparent")
        status_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(status_frame, text="Status:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.status_label = ctk.CTkLabel(status_frame, text="Ready", font=ctk.CTkFont(size=12), text_color="#10b981")
        self.status_label.pack(side="right")
        
        # Accuracy
        acc_frame = ctk.CTkFrame(stats_panel, fg_color="transparent")
        acc_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(acc_frame, text="Accuracy:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.acc_label = ctk.CTkLabel(acc_frame, text="—", font=ctk.CTkFont(size=12, weight="bold"), text_color="#06b6d4")
        self.acc_label.pack(side="right")
        
        # Loss
        loss_frame = ctk.CTkFrame(stats_panel, fg_color="transparent")
        loss_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(loss_frame, text="Loss:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.loss_label = ctk.CTkLabel(loss_frame, text="—", font=ctk.CTkFont(size=12, weight="bold"), text_color="#a855f7")
        self.loss_label.pack(side="right")
        
        # Weights
        weights_frame = ctk.CTkFrame(stats_panel, fg_color="transparent")
        weights_frame.pack(fill="x", padx=15, pady=(5, 15))
        ctk.CTkLabel(weights_frame, text="Weights:", font=ctk.CTkFont(size=12)).pack(side="left")
        self.weights_label = ctk.CTkLabel(weights_frame, text="—", font=ctk.CTkFont(size=11), text_color="gray")
        self.weights_label.pack(side="right")
        
        # Bind slider events
        lr_slider.configure(command=self.update_lr_label)
        epochs_slider.configure(command=self.update_epochs_label)
    
    def create_visualization_panel(self, parent):
        """Create right panel with visualizations"""
        right_panel = ctk.CTkFrame(parent, corner_radius=15, fg_color="#1e1e3a")
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=10)
        
        # Tab view
        tab_view = ctk.CTkTabview(right_panel, corner_radius=10)
        tab_view.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Visualization tab
        viz_tab = tab_view.add("🎨 Visualization")
        self.create_network_viz(viz_tab)
        
        # Theory tab
        theory_tab = tab_view.add("📖 Theory")
        self.create_theory_tab(theory_tab)
        
        # Quiz tab
        quiz_tab = tab_view.add("📝 Quiz")
        self.create_quiz_tab(quiz_tab)
    
    def create_network_viz(self, parent):
        """Create neural network visualization"""
        # Network diagram
        viz_frame = ctk.CTkFrame(parent, fg_color="transparent")
        viz_frame.pack(fill="both", expand=True, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(6, 4), facecolor='#1e1e3a')
        self.ax.set_facecolor('#1e1e3a')
        self.draw_network()
        
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Results panel
        self.results_frame = ctk.CTkFrame(parent, corner_radius=10, fg_color="#2d2d5e")
        self.results_frame.pack(fill="x", pady=10)
    
    def draw_network(self):
        """Draw neural network architecture"""
        self.ax.clear()
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.axis('off')
        
        # Input layer
        input_positions = [(2, 7), (2, 5), (2, 3)]
        # Output layer
        output_positions = [(8, 5)]
        
        # Draw connections
        for inp in input_positions:
            for out in output_positions:
                self.ax.plot([inp[0], out[0]], [inp[1], out[1]], 'w-', alpha=0.3, linewidth=1)
        
        # Draw input neurons
        for x, y in input_positions:
            circle = plt.Circle((x, y), 0.5, color='#06b6d4', ec='white', linewidth=2)
            self.ax.add_patch(circle)
            self.ax.text(x, y, 'x', ha='center', va='center', color='white', fontsize=10, weight='bold')
        
        # Draw output neuron
        x, y = output_positions[0]
        circle = plt.Circle((x, y), 0.6, color='#a855f7', ec='white', linewidth=2)
        self.ax.add_patch(circle)
        self.ax.text(x, y, 'y', ha='center', va='center', color='white', fontsize=10, weight='bold')
        
        # Add labels
        self.ax.text(2, 8.5, 'Input Layer', ha='center', color='#06b6d4', fontsize=10, weight='bold')
        self.ax.text(8, 6.5, 'Output Layer', ha='center', color='#a855f7', fontsize=10, weight='bold')
        
        self.canvas.draw()
    
    def create_theory_tab(self, parent):
        """Create theory/educational content"""
        theory_text = """
        📖 Perceptron Theory
        
        The perceptron is the simplest type of artificial neural network,
        invented by Frank Rosenblatt in 1958.
        
        🔬 McCulloch-Pitts Neuron (1943):
        First mathematical model of a biological neuron.
        
        📐 Mathematical Equation:
        y = f(∑(wᵢ × xᵢ) + b)
        
        🎯 Learning Rule (Hebbian Learning):
        Δw = η × (y_target - y_pred) × x
        
        ⚙️ Activation Functions:
        • Step Function
        • Sigmoid
        • Tanh
        
        ⚠️ Limitations:
        • Cannot solve XOR problem
        • Only works for linearly separable data
        • Single layer limits complexity
        
        💡 Key Insight: The perceptron convergence theorem
        guarantees learning for linearly separable patterns!
        """
        
        text_widget = ctk.CTkTextbox(parent, font=ctk.CTkFont(size=13), wrap="word")
        text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        text_widget.insert("1.0", theory_text)
        text_widget.configure(state="disabled")
    
    def create_quiz_tab(self, parent):
        """Create interactive quiz"""
        quiz_frame = ctk.CTkFrame(parent, fg_color="transparent")
        quiz_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        questions = [
            ("1. What is the output of AND gate for inputs (1,1)?", ["0", "1"], "1"),
            ("2. What does the learning rate control?", ["Number of neurons", "Size of weight updates"], "Size of weight updates"),
            ("3. Which problem cannot be solved by a single perceptron?", ["AND", "OR", "XOR"], "XOR")
        ]
        
        self.quiz_vars = []
        
        for i, (q, options, correct) in enumerate(questions):
            q_frame = ctk.CTkFrame(quiz_frame, fg_color="transparent")
            q_frame.pack(fill="x", pady=10)
            
            ctk.CTkLabel(q_frame, text=q, font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w")
            
            var = ctk.StringVar(value="")
            self.quiz_vars.append((var, correct))
            
            for opt in options:
                radio = ctk.CTkRadioButton(q_frame, text=opt, variable=var, value=opt)
                radio.pack(anchor="w", padx=20)
        
        submit_btn = ctk.CTkButton(quiz_frame, text="Submit Answers", command=self.check_quiz, height=40)
        submit_btn.pack(pady=20)
        
        self.quiz_result = ctk.CTkLabel(quiz_frame, text="", font=ctk.CTkFont(size=13))
        self.quiz_result.pack()
    
    def check_quiz(self):
        """Check quiz answers"""
        score = 0
        for var, correct in self.quiz_vars:
            if var.get() == correct:
                score += 1
        
        result_text = f"✅ Score: {score}/{len(self.quiz_vars)} - "
        if score == len(self.quiz_vars):
            result_text += "Perfect! 🎉"
        elif score >= len(self.quiz_vars)//2:
            result_text += "Good job! 👍"
        else:
            result_text += "Try again! 📚"
        
        self.quiz_result.configure(text=result_text, text_color="#10b981" if score >= 2 else "#f59e0b")
    
    def update_lr_label(self, value):
        """Update learning rate label"""
        self.lr_label.configure(text=f"{float(value):.2f}")
    
    def update_epochs_label(self, value):
        """Update epochs label"""
        self.epochs_label.configure(text=str(int(value)))
    
    def start_training(self):
        """Start training the perceptron"""
        if self.training_active:
            return
        
        self.training_active = True
        self.train_btn.configure(text="⏳ Training...", state="disabled")
        self.status_label.configure(text="Training...", text_color="#f59e0b")
        
        # Get training parameters
        dataset_name = self.dataset_var.get()
        learning_rate = self.lr_var.get()
        epochs = self.epochs_var.get()
        
        data = self.datasets[dataset_name]
        
        # Create and train perceptron
        self.perceptron = Perceptron(2, learning_rate)
        
        def training_thread():
            def update_callback(epoch, loss, accuracy):
                self.after(0, lambda: self.update_stats(epoch, loss, accuracy, epochs))
            
            results = self.perceptron.train(data['X'], data['y'], epochs, update_callback)
            
            self.after(0, lambda: self.training_complete(results))
        
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()
    
    def update_stats(self, epoch, loss, accuracy, total_epochs):
        """Update training statistics in real-time"""
        self.acc_label.configure(text=f"{accuracy:.1f}%")
        self.loss_label.configure(text=f"{loss:.4f}")
        self.weights_label.configure(text=f"{self.perceptron.weights[0]:.3f}, {self.perceptron.weights[1]:.3f}")
        self.status_label.configure(text=f"Epoch {epoch+1}/{total_epochs}", text_color="#f59e0b")
    
    def training_complete(self, results):
        """Handle training completion"""
        self.training_active = False
        self.train_btn.configure(text="🚀 Start Training",