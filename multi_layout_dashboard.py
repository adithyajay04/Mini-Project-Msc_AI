import customtkinter as ctk
import subprocess
import threading

# Modern UI setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ScalabilityDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Traffic Scalability Benchmarking Suite")
        self.geometry("800x550")

        self.label = ctk.CTkLabel(self, text="Select Grid Layouts to Compare", font=("Arial", 20, "bold"))
        self.label.pack(pady=(20, 10))

        # --- NEW: Simulation Mode Toggle ---
        self.mode_var = ctk.StringVar(value="ai") # Default to AI
        
        self.mode_frame = ctk.CTkFrame(self)
        self.mode_frame.pack(pady=10, padx=20, fill="x")
        
        self.mode_label = ctk.CTkLabel(self.mode_frame, text="Simulation Mode:", font=("Arial", 14, "bold"))
        self.mode_label.pack(side="left", padx=20, pady=10)
        
        self.radio_ai = ctk.CTkRadioButton(self.mode_frame, text="MAPPO AI Agent", variable=self.mode_var, value="ai")
        self.radio_ai.pack(side="left", padx=20, pady=10)
        
        self.radio_fixed = ctk.CTkRadioButton(self.mode_frame, text="Standard Fixed Timer (Baseline)", variable=self.mode_var, value="fixed")
        self.radio_fixed.pack(side="left", padx=20, pady=10)
        # -----------------------------------

        # Dashboard layout
        self.btn_frame = ctk.CTkFrame(self)
        self.btn_frame.pack(pady=20, padx=20, fill="both", expand=True)

        self.add_run_button("2x2 Grid", "grid2x2.net.xml", 8813)
        self.add_run_button("3x3 Grid", "grid3x3.net.xml", 8814)
        self.add_run_button("4x4 Grid", "grid4x4.net.xml", 8816)
        self.add_run_button("5x5 Grid", "grid5x5.net.xml", 8815)
        self.add_run_button("Kottayam City", "kottayam.net.xml", 8817)

    def add_run_button(self, name, net_file, port):
        btn = ctk.CTkButton(self.btn_frame, text=f"Run {name}", 
                            command=lambda: self.run_simulation(net_file, port))
        btn.pack(pady=10, padx=20)

    def run_simulation(self, net, port):
        # We launch the simulation in a background thread to prevent UI freezing
        threading.Thread(target=self.launch_process, args=(net, port)).start()

    def launch_process(self, net, port):
        # Pass the selected mode along with the map and port
        selected_mode = self.mode_var.get()
        cmd = ["python", "run_instance.py", "--net", net, "--port", str(port), "--mode", selected_mode]
        subprocess.Popen(cmd)

if __name__ == "__main__":
    app = ScalabilityDashboard()
    app.mainloop()