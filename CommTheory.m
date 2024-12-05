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

