load("DREAMER.mat"); % load DREAMER dataset

% Basic Vars
samplingRate = DREAMER.EEG_SamplingRate; % retrieve sampling rate
windowSize = 256; % window size for each data segment in PSD (power spectral density) calculation
overlap = 128; % overlap between segments

freqRanges = [
    0.5, 4;    % Delta band: 0.5 - 4Hz
    4, 8;      % Theta band: 4 - 8 Hz
    8, 13;     % Alpha band: 8 - 13 Hz
    13, 20;    % Beta band: 13 - 30 Hz
    30, 100;   % Gamma band: 30 - 100 Hz
];

% compute all features and store in .mat file
features_score = zeros(DREAMER.noOfSubjects*DREAMER.noOfVideoSequences, 73);

for i = 1:DREAMER.noOfSubjects
    for j = 1:DREAMER.noOfVideoSequences
        % baseline feature extraction (last 60s)
        EEG_baseline = DREAMER.Data{i}.EEG.baseline{j};
        baseline_last_60s = EEG_baseline(end-samplingRate*60+1:end, :);
        [psd, freq] = getPSD(baseline_last_60s, windowSize, overlap, samplingRate);
        baseline_features = extractEEGFeatures(freq, psd, freqRanges);

        % stimuli feature extraction (last 60s)
        EEG_stimuli = DREAMER.Data{i}.EEG.stimuli{j};
        stimuli_last_60s = EEG_stimuli(end-samplingRate*60+1:end,:);
        [psd, freq] = getPSD(stimuli_last_60s, windowSize, overlap, samplingRate);
        stimuli_features = extractEEGFeatures(freq, psd, freqRanges);
        stimuli_features_normalized = stimuli_features-baseline_features; % use subtraction because its log values

        features_score((i-1)*DREAMER.noOfVideoSequences+j,1:70) = reshape(stimuli_features_normalized.', 1, []);
    end
end

disp(features_score)

% save arousal score at the end column for testing
for i = 1:DREAMER.noOfSubjects
    scoreValence = DREAMER.Data{i}.ScoreValence;
    scoreArousal = DREAMER.Data{i}.ScoreArousal;
    scoreDominance = DREAMER.Data{i}.ScoreDominance;
    for j = 1:DREAMER.noOfVideoSequences
        features_score((i-1)*DREAMER.noOfVideoSequences+j, 71) = scoreValence(j);
        features_score((i-1)*DREAMER.noOfVideoSequences+j, 72) = scoreArousal(j);
        features_score((i-1)*DREAMER.noOfVideoSequences+j, 73) = scoreDominance(j);
    end
end

save('features_score.mat', "features_score");

% Functions
function [psd, freq] = getPSD(eegSignal, windowSize, overlap, samplingRate)
    [psd, freq] = pwelch(eegSignal, windowSize, overlap, windowSize, samplingRate); % use Welch's method to calculate PSD
end

function features = extractEEGFeatures(freq, psd, freqRanges)
    features = zeros(3, 14); % init features with 3 bands and 14 channels
    for i = 1:size(freqRanges)
        curBand = freqRanges(i, :);
        features(i, :) = trapz(psd(freq >= curBand(1) & freq <= curBand(2), :));
    end
    features = 10*log10(features);
end