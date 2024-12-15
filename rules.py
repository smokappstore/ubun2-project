import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


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
        self.modulo_prediccion = self._build_prediction_model(self.state_dim, self.hidden_dim, self.output_dim, self.activation_fn)
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
        return activations.get(name.lower(), nn.ReLU()) # Default to ReLU

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
