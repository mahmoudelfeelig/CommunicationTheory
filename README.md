# 📡 CommunicationTheory

A MATLAB-based project that simulates the entire pipeline of digital signal compression and transmission, incorporating quantization, encoding, channel noise simulation, and signal reconstruction. It includes uniform and μ-law quantization, Huffman coding, error modeling through a Binary Symmetric Channel (BSC), and comparative visualization of signal integrity.

This project provides a practical hands-on demonstration of the concepts found in digital communication and compression theory courses.

---

## 📦 Features

- 🎚️ Signal sampling at multiple rates
- 📏 Uniform and μ-law quantization
- 📊 Quantization error analysis (MAE, variance)
- 🔈 SQNR (Signal-to-Quantization-Noise Ratio) computation
- 🧠 Huffman encoding & decoding
- 🧪 Binary Symmetric Channel (BSC) noise simulation
- 📉 Compression rate calculation
- 📈 Signal reconstruction and correlation analysis
- 📷 Multiple plots comparing input vs output signals

---

## 🧠 Concepts Demonstrated

- Digital signal sampling and quantization
- Non-uniform quantization with μ-law compression
- Lossy compression and SQNR evaluation
- Huffman coding for entropy-based compression
- Bit-flip error modeling via BSC
- Visual and statistical comparison of original and reconstructed signals

---

## 🛠️ Requirements

- MATLAB R2021a or later (or GNU Octave with Signal Toolbox support)
- No additional toolboxes needed (uses built-in functions)

---

## 🚀 How to Run

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mahmoudelfeelig/CommunicationTheory.git
   cd CommunicationTheory
   ```

2. **Open the main script in MATLAB:**

   ```matlab
   open CommunicationTheory.m
   ```

3. **Run the script:**  
   Press **Run** or type in the console:

   ```matlab
   CommunicationTheory
   ```

4. **View the results:**  
   - Multiple plots will appear for sampled signals, quantization analysis, and error comparisons.
   - Compression rate and correlation stats will be printed to the MATLAB console.

---

## 📊 Sample Output Metrics

- **MAE & Variance** for different L values
- **SQNR (in dB)** vs quantization level
- **Compression rate** in %
- **Cross-correlation** between original and reconstructed signals

---

## 📚 Educational Goals

This project is designed to:

- Reinforce understanding of signal quantization and compression
- Visualize the trade-off between quantization level and fidelity
- Simulate real-world transmission errors and noise resilience
- Illustrate end-to-end digital communication system behavior

---

## 📖 License

This project is open-source and distributed under the [MIT License](LICENSE).
