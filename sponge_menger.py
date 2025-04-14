# Esponja de Menger en Python usando Processing.py
# Para ejecutar este código necesitas Processing.py: https://py.processing.org/

# Variables globales
S = 200  # Tamaño de la esponja de Menger
n = 3    # Nivel de recursión (n=3 genera 8000 cubos, manejable en WEBGL)

def setup():
    # Crear un lienzo 3D con WEBGL
    global S, n
    size(600, 600, P3D)

def draw():
    # Fondo negro
    background(0)
    
    # Mover al centro del lienzo para manipulaciones 3D
    translate(width/2, height/2, 0)
    
    # Permitir control de órbita con el ratón (P3D no tiene orbitControl nativo)
    # En su lugar podemos usar las funciones de transformación para simular rotación
    
    # Iluminación ambiental para visibilidad básica
    ambientLight(50, 50, 50)
    
    # Calcular la posición de la luz orbitante
    theta = frameCount * 0.01  # Ángulo que aumenta con el tiempo
    lightPosX = 300 * cos(theta)  # Radio de órbita: 300
    lightPosY = 300 * sin(theta)
    lightPosZ = 100  # Altura fija de la luz
    pointLight(255, 255, 255, lightPosX, lightPosY, lightPosZ)  # Luz blanca
    
    # Aplicar rotaciones a la esponja
    rotateY(theta)        # Rotación en el eje Y
    rotateX(theta * 0.5)  # Rotación más lenta en el eje X
    
    # Dibujar la esponja de Menger en el origen con tamaño S
    menger(n, 0, 0, 0, S)

# Función recursiva para generar la esponja de Menger
def menger(n, x, y, z, s):
    if n == 0:
        # Caso base: dibujar un cubo
        drawCube(x, y, z, s)
    else:
        # Dividir el cubo en subcubos de tamaño s/3
        ss = s / 3
        # Iterar sobre una cuadrícula 3x3x3
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Contar cuántos índices son 1 (centros a eliminar)
                    count = (1 if i == 1 else 0) + (1 if j == 1 else 0) + (1 if k == 1 else 0)
                    if count < 2:
                        # Dibujar subcubos solo si no son centros (menos de 2 índices son 1)
                        menger(n - 1, x + (i - 1) * ss, y + (j - 1) * ss, z + (k - 1) * ss, ss)

# Función para dibujar un cubo con color y reflejo
def drawCube(x, y, z, s):
    pushMatrix()  # Guardar el estado de transformación
    translate(x, y, z)  # Mover a la posición del cubo
    
    # Calcular el gradiente de color basado en la posición z
    t = (z + S / 2) / S  # Normalizar z entre 0 y 1
    
    # En Processing.py, tenemos que manejar el lerpColor de manera diferente
    r = lerp(0, 255, t)
    g = 0
    b = lerp(255, 0, t)
    
    fill(r, g, b)  # Aplicar el color: azul a rojo
    
    # Material especular para reflejos brillantes
    specular(255, 255, 255)  # Color especular blanco
    shininess(32)  # Nivel de brillo
    
    box(s)  # Dibujar el cubo de tamaño s
    popMatrix()  # Restaurar el estado

# Función auxiliar para interpolación lineal
def lerp(start, stop, amt):
    return start + (stop - start) * amt