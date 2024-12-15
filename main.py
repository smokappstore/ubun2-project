import random
import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt


# Modelo de Actor-Crítico para A2C
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


# Módulo de predicción
class ModuloPrediccion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ModuloPrediccion, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.rnn(x)
        return self.fc(hn[-1])


# Aplicación principal
class A2CApplication:
    def __init__(self, title="A2C Agent in Binary Object Environment"):
        # Interfaz de usuario
        self.title = title
        self.initialize_ui()

        # Inicializar módulos
        self.state_dim = 1
        self.action_dim = 4
        self.hidden_dim = 128
        self.actor_critic = ActorCriticA2C(self.state_dim, self.action_dim, self.hidden_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.001)

        # Módulo de predicción
        self.modulo_prediccion = ModuloPrediccion(self.state_dim, self.hidden_dim, 1)
        self.optimizer_prediccion = optim.Adam(self.modulo_prediccion.parameters(), lr=0.001)

        # Experiencias
        self.experiences = {"log_probs": [], "values": [], "rewards": []}

    def initialize_ui(self):
        """Inicializa la interfaz gráfica."""
        self.root = tk.Tk()
        self.root.title(self.title)

        # Sección de salida
        self.feedback_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=60, height=20)
        self.feedback_text.grid(column=0, row=0, padx=10, pady=10)

        # Botones
        ttk.Button(self.root, text="Entrenar", command=self.train_agent).grid(column=0, row=1, padx=10, pady=5)

    def store_experience(self, state, action, reward):
        """Almacena las experiencias de entrenamiento."""
        state_tensor = torch.tensor([state], dtype=torch.float32)
        action_probs, state_value = self.actor_critic(state_tensor)
        log_prob = torch.log(action_probs.squeeze(0)[action])
        self.experiences["log_probs"].append(log_prob)
        self.experiences["values"].append(state_value)
        self.experiences["rewards"].append(reward)

    def select_action(self, state):
        """Selecciona una acción basada en la política actual."""
        state_tensor = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            action_probs, _ = self.actor_critic(state_tensor)
        return torch.multinomial(action_probs, 1).item()

    def calculate_discounted_rewards(self, rewards, gamma):
        """Calcula las recompensas descontadas."""
        R = 0
        discounted = []
        for reward in reversed(rewards):
            R = reward + gamma * R
            discounted.insert(0, R)
        return torch.tensor(discounted, dtype=torch.float32)

    def train_agent(self):
        """Entrena al agente A2C."""
        for epoch in range(500):
            state = random.randint(0, 3)  # Simula un estado inicial
            action = self.select_action(state)
            reward = random.uniform(-1, 1)  # Simula un reward aleatorio
            self.store_experience(state, action, reward)

            # Entrenamiento del Actor-Crítico
            discounted_rewards = self.calculate_discounted_rewards(self.experiences["rewards"], 0.99)
            values = torch.cat(self.experiences["values"])
            log_probs = torch.stack(self.experiences["log_probs"])

            advantages = discounted_rewards - values.squeeze()
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()

            self.optimizer.zero_grad()
            (actor_loss + critic_loss).backward()
            self.optimizer.step()

            # Salida del entrenamiento
            if (epoch + 1) % 100 == 0:
                self.feedback_text.insert(tk.END, f"Epoch {epoch + 1}: Loss: {actor_loss + critic_loss:.4f}\n")
                self.feedback_text.see(tk.END)

        print("Entrenamiento completado.")
        self.root.mainloop()


class MiAgenteVisual:
    def __init__(self, environment: Any, gamma: float = 0.99):
        self.environment = environment
        self.gamma = gamma
        self.trajectory = []
        self.reward_storage = []

    def _record_trajectory(self, state: Any):
        self.trajectory.append(state)

    def plot_trajectory_3d(self):
        if len(self.trajectory) == 0:
            print("No trajectory data available.")
            return

        if len(self.trajectory[0]) < 3:
            print("Trajectory data must have x, y, z components.")
            return

        x, y, z = zip(*self.trajectory)
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x, y, z, label='Agent Trajectory', marker='o')
        ax.set_title('3D Trajectory of Agent')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')
        plt.legend()
        plt.show()

    def calculate_return(self):
        R = 0
        returns = []
        for reward in reversed(self.reward_storage):
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def train_agent(self, total_epochs=500):
        for epoch in range(total_epochs):
            self.reward_storage = []
            initial_state = self.environment.get_initial_state()
            self._record_trajectory(initial_state)

            terminated = False
            while not terminated:
                action = random.choice([0, 1, 2, 3])
                next_state, reward, terminated = self.environment.execute_action(action)
                self._record_trajectory(next_state)
                self.reward_storage.append(reward)

            returns = self.calculate_return()
            print(f"Epoch {epoch + 1}: Total Reward: {sum(self.reward_storage):.2f}")

        self.plot_trajectory_3d()
        print("Training completed.")


class A2CApp:
    def __init__(self):
        self.A2CAgent = self.A2CAgent
        self.main_window = tk.Tk()
        self.main_window.title("A2C - Visualización, Ajuste y Prueba")
        self.main_window.geometry("900x600")

        # Parámetros iniciales del sistema
        self.learning_rate = tk.DoubleVar(value=0.01)  # Se puede ajustar dinámicamente
        self.hidden_dim = tk.IntVar(value=128)  # Tamaño de las capas densas
        self.metrics = {"trajectory": [], "entropy": [], "coherence": [], "quality": []}

    class A2CAgent:
        def __init__(self, environment, state_dim, action_dim, hidden_dim, gamma=0.99, activation_fn="relu",
                     output_dim=None):
            self.environment = environment
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.GAMMA = gamma
            self.output_dim = output_dim if output_dim else state_dim  # Por defecto, igual a state_dim si no se define
            self.activation_fn = self._get_activation_fn(activation_fn)

            # Modelo Actor-Crítico
            self.actor_critic = self._build_model()
            self.optimizer_actor_critic = optim.Adam(self.actor_critic.parameters(), lr=0.001)

            # Módulo de predicción
            self.modulo_prediccion = self._build_prediction_model(self.state_dim, self.hidden_dim, self.output_dim,
                                                                  self.activation_fn
                                                                  )
            self.optimizer_prediccion = optim.Adam(self.modulo_prediccion.parameters(), lr=0.001)

            # Experiencias y métricas
            self.experiences = {"log_probs": [], "values": [], "rewards": []}
            self.metrics = {"trajectory": [], "entropy": [], "coherence": [], "quality": []}

            # Configuración para gráficas
            self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 12))
            self.canvas = None

        def _get_activation_fn(self, name):
            """Helper to retrieve activation function."""
            activations = {"relu": nn.ReLU(),
                           "sigmoid": nn.Sigmoid(),
                           "tanh": nn.Tanh(),
                           "leaky_relu": nn.LeakyReLU(),
                           "elu": nn.ELU(),
                           "gelu": nn.GELU(),
                           }
            return activations.get(name.lower(), nn.ReLU())  # Default to ReLU

        def _build_model(self):
            class ActorCritic(nn.Module):
                def __init__(self, input_dim, action_dim, hidden_dim):
                    super(ActorCritic, self).__init__()
                    self.fc = nn.Linear(input_dim, hidden_dim)
                    self.policy = nn.Linear(hidden_dim, action_dim)
                    self.value = nn.Linear(hidden_dim, 1)

                def forward(self, state):
                    x = torch.relu(self.fc(state))
                    policy_logits = self.policy(x)
                    value = self.value(x)
                    return torch.softmax(policy_logits, dim=-1), value

            return ActorCritic(self.state_dim, self.action_dim, self.hidden_dim)

        def _build_prediction_model(self, input_dim, hidden_dim, output_dim, activation_fn):
            class PredictionModel(nn.Module):
                def __init__(self, input_dim, hidden_dim, output_dim, activation_fn):
                    super(PredictionModel, self).__init__()
                    self.fc = nn.Linear(input_dim, hidden_dim)
                    self.activation = activation_fn
                    self.out = nn.Linear(hidden_dim, output_dim)

                def forward(self, state):
                    x = self.activation(self.fc(state))
                    return self.out(x)

            return PredictionModel(input_dim, hidden_dim, output_dim, activation_fn)

        def calculate_discounted_rewards(self, rewards, gamma):
            R = 0
            discounted = []
            for reward in reversed(rewards):
                R = reward + gamma * R
                discounted.insert(0, R)
            return torch.tensor(discounted, dtype=torch.float32)

        def store_experience(self, state, action, reward):
            """Almacena las experiencias."""
            state_tensor = torch.tensor([state], dtype=torch.float32)
            action_probs, state_value = self.actor_critic(state_tensor)
            log_prob = torch.log(action_probs.squeeze(0)[action])
            self.experiences["log_probs"].append(log_prob)
            self.experiences["values"].append(state_value)
            self.experiences["rewards"].append(reward)

        def select_action(self, state):
            """Selecciona una acción basada en la política actual."""
            state_tensor = torch.tensor([state], dtype=torch.float32)
            action_probs, _ = self.actor_critic(state_tensor)
            return torch.multinomial(action_probs, 1).item()

        def update_actor_critic(self, discounted_rewards):
            """Actualiza las redes Actor-Crítico."""
            values = torch.cat(self.experiences["values"])
            log_probs = torch.stack(self.experiences["log_probs"])
            advantages = discounted_rewards - values.squeeze()

            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = advantages.pow(2).mean()

            self.optimizer_actor_critic.zero_grad()
            (actor_loss + critic_loss).backward()
            self.optimizer_actor_critic.step()

        def train_agent(self):
            """Entrena el agente A2C."""
            num_epocas = 500
            for epoca in range(num_epocas):
                # Reiniciar experiencias
                self.experiences["log_probs"].clear()
                self.experiences["values"].clear()
                self.experiences["rewards"].clear()

                # Obtener estado inicial y ejecutar acción
                state_actual = self.environment.get_state()
                action = self.select_action(state_actual)
                new_state, reward, _ = self.environment.execute_action(action)

                # Almacenar experiencia y calcular recompensas descontadas
                self.store_experience(state_actual, action, reward)
                discounted_rewards = self.calculate_discounted_rewards(self.experiences["rewards"], self.GAMMA)

                # Actualizar redes Actor-Crítico
                self.update_actor_critic(discounted_rewards)

                # Entrenar módulo de predicción
                state_tensor = torch.tensor([[state_actual]], dtype=torch.float32)
                futuro_state_tensor = torch.tensor([[new_state]], dtype=torch.float32)
                prediccion_loss = nn.MSELoss()(self.modulo_prediccion(state_tensor), futuro_state_tensor)

                self.optimizer_prediccion.zero_grad()
                prediccion_loss.backward()
                self.optimizer_prediccion.step()

                # Guardar métricas
                entropy = np.random.uniform(0, 1)  # Placeholder
                coherence = np.random.uniform(0, 1)  # Placeholder
                quality = np.random.uniform(0, 1)  # Placeholder

                self.metrics["trajectory"].append(action)
                self.metrics["entropy"].append(entropy)
                self.metrics["coherence"].append(coherence)
                self.metrics["quality"].append(quality)

                # Actualizar gráficas
                self.update_graphs()

                if (epoca + 1) % 100 == 0:
                    print(f"Epoca: {epoca + 1}, Prediccion Loss: {prediccion_loss:.4f}")

            print("Entrenamiento completado.")

        def update_graphs(self):
            """Actualizar los gráficos con base en los datos actuales."""
            metrics = ["trajectory", "entropy", "coherence", "quality"]
            colors = ["blue", "orange", "green", "purple"]
            labels = ["Trayectoria", "Entropía", "Coherencia", "Calidad"]

            for i, metric in enumerate(metrics):
                self.axs[i].cla()
                self.axs[i].plot(self.metrics[metric], color=colors[i], label=labels[i])
                self.axs[i].legend()

            self.fig.canvas.draw()

    def _build_model(self, input_dim, action_dim, hidden_dim, activation_fn):
        class ActorCritic(nn.Module):

            def __init__(self, input_dim, action_dim, hidden_dim, activation_fn):
                super(ActorCritic, self).__init__()
                self.fc = nn.Linear(input_dim, hidden_dim)
                self.activation = activation_fn
                self.policy = nn.Linear(hidden_dim, action_dim)
                self.value = nn.Linear(hidden_dim, 1)

            def forward(self, state):
                x = self.activation(self.fc(state))
                policy_logits = self.policy(x)
                value = self.value(x)
                return torch.softmax(policy_logits, dim=-1), value

        return ActorCritic(input_dim, action_dim, hidden_dim, activation_fn())

    def run_test(self):
        """Simular pruebas del sistema y actualizar métricas."""
        self.run_test()
        self.total_epochs = 200
        print(f"{self._build_model(activation_fn=Any, input_dim=Any, action_dim=Any, hidden_dim=Any)}")

        for _ in range(10):  # Simular 10 pasos
            # Simular métricas para la demostración
            state = torch.rand(4)  # Estado aleatorio
            action_probs = self.actor_network(state).detach(None).numpy()
            entropy = -np.sum(action_probs * np.log(action_probs))  # Entropía
            coherence = random.uniform(0.6, 1.0)  # Coherencia simulada
            quality = random.uniform(0, 1)  # Calidad simulada

            # Actualizar métricas almacenadas
            self.metrics["trajectory"].append(np.argmax(action_probs))  # Acción con mayor probabilidad
            self.metrics["entropy"].append(entropy)
            self.metrics["coherence"].append(coherence)
            self.metrics["quality"].append(quality)

            # Actualizar gráficos
            self.update_graphs()

    def run(self):
        """Iniciar el loop principal de la aplicación."""
        self.main_window.mainloop()

    def actor_network(self, state):
        self.__getstate__()
        self.A2CAgent.train_agent(state)

    def update_graphs(self, detach):
        print(f"{A2CApp}")


if __name__ == '__main__':
    class DoubleVar:
        def __init__(self, main_op):
            if __name__ == '__main__':
                main_op.__str__()

main_op = A2CApplication(title="A2C Agent in Binary Object Enviroment")
print(f"{main_op}")

app = A2CApplication
tk.DoubleVar(master=None)
