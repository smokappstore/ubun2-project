import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class MultiWindowApp:
    def __init__(self):
        # Ventana principal
        self.result_label = None
        self.shared_data = None
        self.main_window = tk.Tk()
        self.main_window.title("Ventana principal - Panel de Control")
        self.main_window.geometry("400x300")

        # Botones en la ventana principal
        ttk.Button(self.main_window, text="Abrir Ventana de Resultados", command=self.open_results_window).pack(pady=10)
        ttk.Button(self.main_window, text="Abrir Ventana de Gráficas", command=self.open_graph_window).pack(pady=10)
        ttk.Button(self.main_window, text="Enviar Información", command=self.send_data_to_results).pack(pady=10)

        # Entrada de texto en la ventana principal
        self.command_entry = ttk.Entry(self.main_window, width=30)
        self.command_entry.pack(pady=10)
        self.command_entry.insert(0, 'escribe aqui...')

        # Crear referencias para las otras ventanas (para comunicarse entre ellas)
        self.results_window = None
        self.graph_window = None

        # Variable que usaremos para actualizar las ventanas
        self.shared_data = "Sin datos enviados aún"

    def open_results_window(self):
        if self.results_window is None or not self.results_window.winfo_exists():
            self.results_window = tk.Toplevel(self.main_window)
            self.results_window.title("Ventana de Resultados")
            self.results_window.geometry("400x200")
            self.result_label = ttk.Label(self.results_window, text=self.shared_data, wraplength=300)
            self.result_label.pack(pady=20)

    def open_graph_window(self):
        if self.graph_window is None or not self.graph_window.winfo_exists():
            self.graph_window = tk.Toplevel(self.main_window)
            self.graph_window.title("Ventana de Gráfica 3D")
            self.graph_window.geometry("600x500")
            # Crear gráfico 3D
            fig = Figure(figsize=(5, 4), dpi=100)

            # Datos para la gráfica (personalizable)
            np.random.random(100)
            np.random.random(100)
            np.random.random(100)
            ax = fig.add_subplot(111, projection='3d')

            ax.set_title("Gráfica 3D Ejemplo")
            ax.set_xlabel("Eje X")
            ax.set_ylabel("Eje Y")
            ax.set_zlabel("Eje Z")

            # Incluir gráfica en la ventana usando tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.graph_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def send_data_to_results(self):
        """Envía los datos de la entrada a la ventana de resultados"""
        if self.results_window is None or not self.results_window.winfo_exists():
            messagebox.showerror("Error", "Primero abre la ventana de resultados.")
            return

        # Obtener texto de la entrada y actualizar compartido
        data = self.command_entry.get()
        if data.strip() == "":
            data = "No se escribió nada."
        self.shared_data = f"Datos enviados: {data}"

        # Actualizar etiqueta en la ventana de resultados
        self.result_label.config(text=self.shared_data)

    def run(self):
        """Ejecuta el loop principal de la app"""
        self.main_window.mainloop()


# Correr la aplicación
if __name__ == "__main__":
    app = MultiWindowApp()
    app.run()