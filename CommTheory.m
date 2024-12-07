% Parameters

A = 9; % Signal Amplitude [13001578 --> 1 + 8]
f = 1000; % Frequency in Hz
duration = 0.01; % Signal duration in seconds
T_s = 0.0001; % Sampling interval
t = 0:T_s:duration; % Time vector
L = 32; % Number of quantization levels
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

%% Simulation time :3

% Step 1: Input Signal and Sampling

input_signal = sample_signal(A, f, T_s, duration);

% Step 2: Quantization

[uniform_quantized, uniform_levels] = uniform_quantizer(input_signal, L, A);
mu_quantized = mu_law_quantizer(input_signal, L, mu);

% Step 3: Quantization Error Analysis

L_Values = [4, 8, 16];
mae_Values = [0, 0, 0];
var_Values = [0, 0, 0];
i = 1;

for LValue = L_Values
    [uniform_quantized, uniform_levels] = uniform_quantizer(input_signal, LValue, A);
    [mae_uniform, var_uniform] = quantization_error_analysis(input_signal, uniform_quantized);
    % [mae_uniform, var_uniform] = quantization_error_analysis(input_signal, mu_quantized);
    % disp(['Mean Absolute Error (Uniform): ', num2str(mae_uniform), 'For the value of L:', num2str(LValue)]);
    % disp(['Variance of Error (Uniform): ', num2str(var_uniform), 'For the value of L:', num2str(LValue)]);
    mae_Values(1, i) = mae_uniform;
    var_Values(1, i)= var_uniform;
    i = i+1;
end
figure;
plot(mae_Values, L_Values);
title('Mean values vs L values');
xlabel('Mean Values');
ylabel('L values');

figure;
plot(var_Values, L_Values);
title('Variance values vs L values');
xlabel('Variance Values');
ylabel('L values');

% Step 4: SQNR

% sqnr_uniform = calculate_sqnr(input_signal, uniform_quantized);
% sqnr_uniform = calculate_sqnr(input_signal, mu_quantized);
% disp(['SQNR (Uniform): ', num2str(sqnr_uniform), ' dB']);

sqnr_Values = [0, 0, 0];
i = 1;
for LValue = L_Values
    [uniform_quantized, uniform_levels] = uniform_quantizer(input_signal, LValue, A);
    sqnr_uniform = calculate_sqnr(input_signal, uniform_quantized);
    % sqnr_uniform = calculate_sqnr(input_signal, mu_quantized);
    sqnr_Values(1, i) = sqnr_uniform;
    i = i+1;
end
figure;
plot(sqnr_Values, L_Values);
title('SQNR values vs L values');
xlabel('SQNR Values (db)');
ylabel('L values');

% Step 5: Encoding (Huffman Encoding)

[encoded_signal, huffman_dict] = huffman_encode(uniform_quantized);
% [encoded_signal, huffman_dict] = huffman_encode(mu_quantized);

% Step 6: Simulate a Channel (Noiseless)

% Huffman decoding to reconstruct the signal
decoded_signal = huffman_decode(encoded_signal, huffman_dict);

% Step 7: Signal Comparison

% Rescale decoded signal for comparison
quantization_step = uniform_levels(2) - uniform_levels(1);
reconstructed_signal = decoded_signal * quantization_step;


% Cross-correlation between input and reconstructed signal
correlation = corrcoef(input_signal, reconstructed_signal);
disp(['Cross-correlation: ', num2str(correlation(1, 2))]);

% Plotting Input vs Output
figure;
plot(t, input_signal, 'b', 'DisplayName', 'Original Signal');
hold on;
plot(t, reconstructed_signal, 'r--', 'DisplayName', 'Reconstructed Signal');
legend;
title('Input Signal vs Reconstructed Signal');
xlabel('Time (s)');
ylabel('Amplitude');

% Step 8: Compression Efficiency and Rate

original_bits = length(input_signal) * ceil(log2(L));
compressed_bits = length(encoded_signal);
compression_rate = calculate_compression_rate(original_bits, compressed_bits);
disp(['Compression Rate: ', num2str(compression_rate * 100), '%']);

% Step 9: Error Channel Simulation (Bonus)

bsc_signal = bsc_channel(encoded_signal, p);
try
    noisy_decoded = huffman_decode(double(bsc_signal), huffman_dict);
    noisy_reconstructed_signal = noisy_decoded * quantization_step;
    % Plot noisy reconstructed signal

    % Ensure the lengths match for plotting
    n = length(noisy_reconstructed_signal);
    min_length = min(length(input_signal), n);  % Ensure matching lengths
    t_reconstructed = linspace(0, duration, min_length);  % Adjust time vector length

    % display the cross-correlation in the erroneous :) channel
    % commenting the code because it produces an error when the signals have
    % different lengths
    % normalized_xcorr = xcorr(input_signal, noisy_reconstructed_signal, 'coeff');
    % disp(['Cross-correlation in the erroneous channel: ', num2str(normalized_xcorr(1, 2))]);

    figure;
    plot(t_reconstructed, input_signal(1:min_length), 'b', 'DisplayName', 'Original Signal');
    hold on;
    plot(t_reconstructed, noisy_reconstructed_signal(1:min_length), 'g--', 'DisplayName', 'Noisy Reconstructed Signal');
    legend('Location', 'best')
    title('Input Signal vs Noisy Reconstructed Signal');
    xlabel('Time (s)');
    ylabel('Amplitude');

catch e
    disp('Error during huffman decoding due to an invalid encoded sequence')
    rethrow(e)
end

%% Enhancing Signal Approximation

% one could increase the number of quantization levels but that would be 
% resource-extensive if implemented irl
% more elegantly we could resort to more advanced encoding techniques
% such as arithmetic coding :)