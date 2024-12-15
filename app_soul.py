import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from keras.src.backend import result_type
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# Modelo A2C
class ActorCriticA2C(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim):
        super(ActorCriticA2C, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.policy = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc(state))
        policy_logits = self.policy(x)
        value = self.value(x)
        return torch.softmax(policy_logits, dim=-1), value


# Aplicación principal con ventanas múltiples
class MultiWindowApp:
    def __init__(self):

        self.main_window = tk.Tk()
        self.main_window.title("A2C - Panel de Control")
        self.main_window.geometry("400x300")
        self.result_label= self
        self.shared_data = result_type(dtypes=[])
        self.results_window= self.main_window
        self.graph_window = None

        # Botones principales
        ttk.Button(self.main_window, text="Abrir Ventana de Resultados", command=self.open_results_window).pack(pady=10)
        ttk.Button(self.main_window, text="Abrir Ventana de Gráficas", command=self.open_graph_window).pack(pady=10)
        ttk.Button(self.main_window, text="Entrenar Agente", command=self.train_agent).pack(pady=10)

        # Entrada de texto
        self.command_entry = ttk.Entry(self.main_window, width=30)
        self.command_entry.pack(pady=10)
        self.command_entry.insert(0, "Escribe aquí...")

        # Inicializar A2C
        self.state_dim = 1
        self.action_dim = 4
        self.hidden_dim = 128
        self.actor_critic = ActorCriticA2C(self.state_dim, self.action_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.001)
        self.experiences = {"log_probs": [], "values": [], "rewards": []}

    def open_results_window(self):
        """Abre la ventana de resultados"""
        if self.results_window is None or not self.results_window.winfo_exists():
            self.results_window = tk.Toplevel(self.main_window)
            self.results_window.title("Ventana de Resultados")
            self.results_window.geometry("400x300")

            # Mostrar resultados
            self.result_label = ttk.Label(self.results_window, text=self.shared_data, wraplength=300)
            self.result_label.pack(pady=20)

    def open_graph_window(self):
        """Abre la ventana de gráficas"""
        if self.graph_window is None or not self.graph_window.winfo_exists():
            self.graph_window = tk.Toplevel(self.main_window)
            self.graph_window.title("Ventana de Gráficas")
            self.graph_window.geometry("600x500")

            # Crear gráfica
            fig = Figure(figsize=(5, 4), dpi=100)
            ax = fig.add_subplot(111, projection="3d")
            x, y, z = np.random.random(100), np.random.random(100), np.random.random(100)
            ax.scatter(x, y, z, c=z, cmap="viridis", marker="o")
            ax.set_title("Gráfica 3D")
            ax.set_xlabel("Eje X")
            ax.set_ylabel("Eje Y")
            ax.set_zlabel("Eje Z")

            # Agregar gráfica a la ventana
            canvas = FigureCanvasTkAgg(fig, master=self.graph_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def train_agent(self):
        """Entrena el agente A2C"""
        for epoch in range(100):
            # Simula un estado inicial y una acción
            state = torch.tensor([[np.random.randint(0, 10)]], dtype=torch.float32)
            action_probs, value = self.actor_critic(state)
            action = torch.multinomial(action_probs, 1).item()
            reward = np.random.uniform(-1, 1)

            # Almacenar experiencias
            log_prob = torch.log(action_probs[0, action])
            self.experiences["log_probs"].append(log_prob)
            self.experiences["values"].append(value)
            self.experiences["rewards"].append(reward)

            # Mostrar progreso cada 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}: Acción {action}, Recompensa {reward:.2f}")

        # Muestra el entrenamiento completado
        self.shared_data = "Entrenamiento completado"
        messagebox.showinfo("Entrenamiento", self.shared_data)

    def run(self):
        """Inicia la aplicación"""
        self.main_window.mainloop()


# Ejecutar la aplicación
if __name__ == "__main__":
    app = MultiWindowApp()
    app.run()
