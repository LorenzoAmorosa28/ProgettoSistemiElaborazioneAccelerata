# Background Subtraction in CUDA

![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Google%20Colab-lightgrey.svg)

Questo progetto implementa un algoritmo di **Background Subtraction**, ottimizzato per l'elaborazione di video utilizzando NVIDIA CUDA.

## 🧠 How it works

Il codice elabora il flusso video frame per frame, applicando a ogni pixel una catena di 4 passaggi logici:

1.  **Riduzione del Rumore (Smoothing)**
    Il frame in ingresso viene "pulito" applicando un filtro di sfocatura (Gaussiana 5x5). Questo passaggio è fondamentale per evitare che il normale rumore digitale della telecamera venga scambiato erroneamente per movimento.

2.  **Confronto con lo Storico**
    Ogni pixel viene confrontato con il "modello di sfondo" che l'algoritmo ha memorizzato nel tempo (Media). Si calcola matematicamente quanto il pixel attuale differisce da quello atteso.

3.  **Classificazione (Sfondo o Oggetto?)**
    Il sistema decide se la differenza rilevata è significativa:
    * Se la differenza supera una soglia dinamica (basata sulla Varianza storica), il pixel viene etichettato come **Movimento** (Bianco).
    * Altrimenti, viene considerato parte dello **Sfondo** (Nero).

4.  **Aggiornamento del Modello (Learning)**
    Infine, il "modello di sfondo" viene aggiornato gradualmente integrando il nuovo pixel. Questo permette al sistema di adattarsi automaticamente ai cambiamenti ambientali lenti, come il cambio di illuminazione naturale o ombre che si spostano durante il giorno.
## 📊 Benchmark Prestazionali

Testato su **NVIDIA Tesla T4** (Google Colab) con Video 4K ($3840 \times 2160$).

| Implementazione | Tempo di Esecuzione (per frame) | FPS Teorici |
| :--- | :---: | :---: |
| CPU | ~85.0 ms | ~12 FPS |
| CUDA | ~0.98 ms | >1000 FPS |

*Verificato tramite NVIDIA Nsight Compute (Analisi Roofline).*

## 🛠️ Dipendenze

* **NVIDIA CUDA Toolkit** (v10.0+)
* **OpenCV 4** (Librerie C++)
* **Compilatore C++** (g++ / nvcc)

## 👥 Autori

* **Amorosa Lorenzo**

* **Giannitti Sebastiano**
