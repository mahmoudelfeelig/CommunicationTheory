% Parameters

A = 9; % Signal Amplitude [13001578 --> 1 + 8]
f = 1000; % Frequency in Hz
duration = 0.01; % Signal duration in seconds
T_s = 0.0001; % Sampling interval
t = 0:T_s:duration; % Time vector
% L = 8; % Number of quantization levels
mu = 255; % For μ-law quantization
p = 0.01; % Error probability in Binary Symmetric Channel

%% Sampling

function samples = sample_signal(A, f, T_s, duration)
    t = 0:T_s:duration;  % Time vector
    samples = A * sin(2 * pi * f * t);  % Generate sinusoidal signal
end

T_s_values = [0.001, 0.0005, 0.0001]; % Sampling intervals

for T_s = T_s_values
    samples = sample_signal(A, f, T_s, duration);
    % figure;
    % plot(0:T_s:duration, samples);
    % title(['Sampled Signal with T_s = ', num2str(T_s)]);
    % xlabel('Time (s)');
    % ylabel('Amplitude');
end

%% Quantization

% Uniform
% function [quantized_values, levels] = uniform_quantizer(signal, L, A)
%     levels = linspace(-A, A, L);  % Define L quantization levels
%     delta = levels(2) - levels(1); % Step size
%     quantized_values = round((signal + A) / delta) * delta - A; % Quantize
% end

function [quantized_values, levels] = uniform_quantizer(signal, L, A)
    % Define L quantization levels linearly spaced between -A and A
    levels = linspace(-A, A, L);  
    
    % Quantize the signal: find the closest level to each sample
    quantized_values = zeros(size(signal));  % Initialize output array
    for i = 1:length(signal)
        % Find the closest quantization level
        [~, idx] = min(abs(signal(i) - levels));  % Find closest level index
        quantized_values(i) = levels(idx);  % Assign the quantized value
    end
end

% meow :3
function quantized_values = mu_law_quantizer(signal, L, mu)
    normalized_signal = signal / max(abs(signal)); % Normalize
    compressed_signal = sign(normalized_signal) .* log(1 + mu * abs(normalized_signal)) / log(1 + mu);
    quantized_signal = round((compressed_signal + 1) * (L - 1) / 2); % Uniform quantization
    quantized_values = 2 * quantized_signal / (L - 1) - 1; % Decompress
    quantized_values = quantized_values * max(abs(signal)); % Restore scale
end

%% Quantization Error Analysis

function [mae, var_error] = quantization_error_analysis(signal, quantized_signal)
    error = signal - quantized_signal;
    mae = mean(abs(error)); % Mean Absolute Error
    var_error = var(error); % Variance of Error
end

%% Signal-to-Quantization Noise Ratio (SQNR)

function sqnr = calculate_sqnr(signal, quantized_signal)
    signal_power = mean(signal .^ 2);
    error = signal - quantized_signal;
    noise_power = mean((error) .^ 2);
    sqnr = 10 * log10(signal_power / noise_power);
end

%% Huffman Encoding

function [encoded_signal, dict] = huffman_encode(signal)
    symbols = unique(signal);
    probabilities = histcounts(signal, [symbols, max(symbols) + 1], 'Normalization', 'probability');
    dict = huffmandict(symbols, probabilities);
    encoded_signal = huffmanenco(signal, dict);
end

%% Huffman Decoding

function decoded_signal = huffman_decode(encoded_signal, dict)
    decoded_signal = huffmandeco(encoded_signal, dict);
end

%% Signal Comparison

% no functions here :)

%% Compression Metrics

function compression_rate = calculate_compression_rate(original_bits, compressed_bits)
    compression_rate = (original_bits - compressed_bits) / original_bits;
end

%% Error Channel Simulation

function noisy_signal = bsc_channel(encoded_signal, p)
    noisy_signal = xor(encoded_signal, rand(size(encoded_signal)) < p);
    noisy_signal = double(noisy_signal);
end

