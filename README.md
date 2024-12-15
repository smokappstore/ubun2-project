# CoreMind: IA Proactiva y Cognitiva para Sistemas Seguros

## Descripción del Proyecto

CoreMind es un proyecto de inteligencia artificial (IA) que busca crear un sistema proactivo y cognitivo, capaz de aprender y adaptarse a su entorno de manera autónoma y ética. A diferencia de las IA tradicionales que esperan comandos del usuario, CoreMind utiliza un enfoque basado en:

*   **Honeypots como Sensores:** Simulación de servicios para atraer y estudiar atacantes, y analizar el entorno de manera pasiva.
*   **Detección de Anomalías:** Identificación de comportamientos atípicos en el entorno usando Isolation Forest.
*   **Deducción Lógica:** Capacidad de la IA para razonar y tomar decisiones usando reglas.
*   **Predicción Probabilística:** Uso de modelos para anticipar los cambios del entorno.
*   **Evaluación de la Calidad Ética y Moral:** Uso de redes antagónicas para evaluar acciones.
*   **Aprendizaje Continuo:** Adaptabilidad mediante retroalimentación del entorno y de las interacciones del usuario.

El proyecto se divide en dos versiones:

1.  **CoreMind para Desarrolladores:** Una plataforma flexible para experimentación y desarrollo de la IA.
2.  **CoreMind para Usuarios:** Una interfaz tipo "Tamagotchi" para pruebas y retroalimentación.

## Estructura del Proyecto

El proyecto está organizado en los siguientes módulos principales:

*   `main.py`: Punto de entrada para la aplicación, que integra la lógica de la IA, la interfaz de usuario, y la gestión de las acciones.
*   `moduls.py`: Implementación de la clase `BaseTrainer` y los diferentes modelos de redes neuronales como `ActorCritic`, `ModuloPrediccion`, y la clase de reglas `ModuloAprendizajeReglas`.
*   `security.py`: Define el sistema de seguridad, con la lógica de los honeypots y los detectores de anomalías.
* `training.py` define la lógica para el entrenamiento de todos los modulos.
*  Archivos para la gestion de las emociones:
  *   `emotions.py`:  Gestiona el sistema de emociones del tamagotchi.
  * `interactions.py`: Define funciones de gestion de interacciones
*   Archivos adicionales de la interfaz de usuario:
    *   `multiapp.py`: Define la estructura de la interfaz multi ventana con `tkinter`.
     * `interface.py`:  Define la estructura de la interfaz principal.
*   Archivos adicionales de la logica principal:
    *  `logic.py`: Implementa la lógica principal de los modulos como `ModuloDeduccion`, `ModuloEvaluacion`, `ModuloCalidadInformacion`, etc.
    *  `utils.py`:  Implementa funciones auxiliares para el proyecto.

## Instalación y Ejecución

1.  **Clonar el repositorio:**
    ```bash
    git clone [URL del repositorio]
    cd [nombre del repositorio]
    ```
2.  **Crear un entorno virtual:** (Recomendado)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Instalar dependencias:**
    ```bash
    pip install torch numpy matplotlib scikit-learn
    pip install pillow
    pip install tensorflow
    ```
4. **Ejecutar el Programa:**
   ```bash
   python main.py# app
# app
