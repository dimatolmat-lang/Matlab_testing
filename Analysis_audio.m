%% =====================================================================
%  АНАЛИЗ 4-МИКРОФОННОЙ ЗАПИСИ ТЕСТОВОГО СИГНАЛА (линейная решётка)
%  Версия 3.1 — заменяет предыдущую целиком.
%  Требования: MATLAB R2016b+, Signal Processing Toolbox.
%  Рассчитан на 4 микрофона в линию (N = 4).
%
%  Настроено под текущую установку: база соседей 165 мм,
%  источник по центру на ~4.2 м, запись полного сигнала генератора.
%
%  КОНВЕНЦИЯ ЗНАКА: tau > 0 = старший канал приходит РАНЬШЕ младшего.
%  Угол: 0° = по нормали к центру решётки; +90° = сторона +X (ch3).
%  Если измеренный угол зеркален задаваемому — добавьте после micPos:
%      micPos(:,1) = -micPos(:,1);
%
%  Рекомендуется запускать из папки с именем «Угол_N» — заданный угол
%  подхватится автоматически и попадёт в отчёт.
%% =====================================================================
close all; clear; clc;

%% 0. НАСТРОЙКИ ---------------------------------------------------------
files = {'channel_0_merged.wav','channel_1_merged.wav', ...
         'channel_2_merged.wav','channel_3_merged.wav'};
refCh = 1;                     % опорный канал (столбец 1 = channel_0)

frameSec   = 1.0;              % окно обзора, с
overlapSec = 0.5;
specWinSec = 0.2;              % окно спектральных оценок, с

maxDelaySec = 0.0018;          % окно поиска задержки, +-1.8 мс
                               % (размах решётки 0.66 м -> крайние до 1.92 мс)

freqsOfInterest = [110 500 1000 1900];
fMinAnal = 50;   fMaxAnal = 3000;

% --- геометрия: 4 микрофона в ряд по X, шаг dMic, центр (0,0) ---
dMic   = 0.165;                                % база соседей, м
micPos = [-1.5*dMic 0; -0.5*dMic 0; 0.5*dMic 0; 1.5*dMic 0];
cSnd   = 343;                                  % скорость звука, м/с
tauMaxMs = 0.98*dMic/cSnd*1000;                % физический предел, мс (~471)

saveResults = true;
storePhase  = true;

%% 1. Загрузка и подготовка --------------------------------------------
N = numel(files);
raw = cell(N,1);  fsAll = zeros(N,1);
for i = 1:N
    [x, fs_i] = audioread(files{i});
    if size(x,2) > 1, x = mean(x,2); end
    raw{i} = x(:);   fsAll(i) = fs_i;
    fprintf('ch%d: %g Гц, %.1f с\n', i-1, fs_i, numel(x)/fs_i);
end
if any(fsAll ~= fsAll(1))
    error('Частоты дискретизации файлов различаются: %s', mat2str(fsAll.'));
end
fs = fsAll(1);
minLen = min(cellfun(@numel, raw));
data = zeros(minLen, N);
for i = 1:N, data(:,i) = raw{i}(1:minLen); end
clear raw;
data = data - mean(data,1);
L = size(data,1);

%% 2. Параметры сегментации --------------------------------------------
frameLen  = round(frameSec * fs);
overlap   = round(overlapSec * fs);
step      = frameLen - overlap;
numFrames = floor((L - overlap) / step);
t         = ((0:numFrames-1)*step + frameLen/2) / fs;

specWinLen   = round(specWinSec * fs);
specWin      = hann(specWinLen);
specNoverlap = round(0.5 * specWinLen);
specNfft     = 2^nextpow2(specWinLen);
Fgrid        = (0:specNfft/2)' * fs / specNfft;
nBins        = numel(Fgrid);
valid        = Fgrid > fMinAnal & Fgrid < fMaxAnal;

maxLag  = round(maxDelaySec * fs);
fLagMax = 1/(2*maxDelaySec);                 % ~278 Гц

%% 3. Память под результаты --------------------------------------------
rmsVal   = nan(numFrames, N);   peakVal  = nan(numFrames, N);
delayMs  = nan(numFrames, N-1); polarity = nan(numFrames, N-1);
cohMed   = nan(numFrames, N-1);
magAt    = nan(numFrames, N-1, numel(freqsOfInterest));
phaseAt  = nan(numFrames, N-1, numel(freqsOfInterest));
cohSum   = zeros(nBins, N-1);   cohCnt = 0;
if storePhase, phaseMat = nan(numFrames, nBins); end

%% 4. Основной цикл ------------------------------------------------------
for k = 1:numFrames
    idx = (k-1)*step + (1:frameLen);
    seg = data(idx,:);
    if max(sqrt(mean(seg.^2,1))) < 1e-4
        continue;                      % тишина — остаются NaN
    end
    rmsVal(k,:)  = sqrt(mean(seg.^2,1));
    peakVal(k,:) = max(abs(seg),[],1);

    Pxx = nan(nBins, N);
    for ch = 1:N
        [Pc, ~] = pwelch(seg(:,ch), specWin, specNoverlap, specNfft, fs);
        Pxx(:,ch) = Pc;
    end

    x = seg(:,refCh);
    for ch = 1:N
        if ch == refCh, continue; end
        y = seg(:,ch);
        j = ch - (ch > refCh);

        [delayMs(k,j), polarity(k,j)] = gccphat_lag(x, y, fs, maxLag);

        [Pxy, ~] = cpsd(x, y, specWin, specNoverlap, specNfft, fs);
        H   = Pxy ./ Pxx(:,refCh);
        coh = abs(Pxy).^2 ./ (Pxx(:,refCh) .* Pxx(:,ch));
        cohSum(:,j) = cohSum(:,j) + coh;
        cohMed(k,j) = median(coh(valid));

        for fi = 1:numel(freqsOfInterest)
            [~, idxF] = min(abs(Fgrid - freqsOfInterest(fi)));
            magAt(k,j,fi)   = abs(H(idxF));
            phaseAt(k,j,fi) = angle(H(idxF));
        end
        if storePhase && ch == 2
            phaseMat(k,:) = angle(H);
        end
    end
    cohCnt = cohCnt + 1;
end

%% 5. Обзорные графики ---------------------------------------------------
figure('Name','RMS по каналам');
plot(t, 20*log10(rmsVal + eps)); grid on;
xlabel('Время, с'); ylabel('RMS, dBFS'); title('Уровень RMS');
legend(compose('ch%d',0:N-1));

figure('Name','Задержка (обзор; на тонах GCC ненадёжен)');
plot(t, delayMs); grid on;
xlabel('Время, с'); ylabel('Задержка, мс');
title('GCC-PHAT относительно опорного (>0 = канал раньше)');
legend(compose('ch%d-ch0',1:N-1));

figure('Name','Когерентность (медиана по полосе)');
plot(t, cohMed); ylim([0 1]); grid on;
xlabel('Время, с'); ylabel('Медианная когерентность');
legend(compose('ch%d-ch0',1:N-1));

figure('Name','Когерентность vs частота');
plot(Fgrid, cohSum / max(cohCnt,1)); ylim([0 1]); grid on;
xlim([fMinAnal fMaxAnal]);
xlabel('Частота, Гц'); ylabel('Средняя когерентность');
legend(compose('ch%d-ch0',1:N-1));

if storePhase
    figure('Name','Разность фаз ch1-ch0');
    imagesc(t, Fgrid, phaseMat.'); axis xy; colorbar;
    xlabel('Время, с'); ylabel('Частота, Гц');
    title('Разность фаз, рад'); ylim([fMinAnal fMaxAnal]);
end

v = rmsVal(:,refCh);
if any(~isnan(v))
    [~, kLoud] = max(v);
    idx = (kLoud-1)*step + (1:frameLen);
    figure('Name','Вектоскоп ch0-ch1');
    plot(data(idx,refCh), data(idx,2)); axis equal; grid on;
    xlabel('ch0'); ylabel('ch1');
    title(sprintf('Вектоскоп, t = %.1f с', t(kLoud)));
end

figure('Name','АЧХ и ФЧХ на выбранных частотах (ch1-ch0)');
colsF = lines(numel(freqsOfInterest));
for fi = 1:numel(freqsOfInterest)
    subplot(2,1,1);
    plot(t, 20*log10(magAt(:,1,fi)+eps), 'Color', colsF(fi,:), ...
         'DisplayName', sprintf('%d Гц', freqsOfInterest(fi))); hold on;
    subplot(2,1,2);
    plot(t, phaseAt(:,1,fi), 'Color', colsF(fi,:), ...
         'DisplayName', sprintf('%d Гц', freqsOfInterest(fi))); hold on;
end
subplot(2,1,1); grid on; ylabel('|H|, дБ');
title('Отношение амплитуд ch1/ch0'); legend show; hold off;
subplot(2,1,2); grid on; ylabel('рад'); xlabel('Время, с');
legend show; hold off;

%% 6. ДЕТЕКЦИЯ СЕГМЕНТОВ (тон/пауза) -------------------------------------
hopE = round(0.05*fs);   winE = round(0.2*fs);
nE = floor((L-winE)/hopE) + 1;
env = zeros(nE,1);
for e = 1:nE
    idx = (e-1)*hopE + (0:winE-1) + 1;
    env(e) = sqrt(mean(data(idx,refCh).^2));
end
le = log10(max(env, 1e-10));   mu = [min(le) max(le)];
for it = 1:100
    lab = abs(le - mu(2)) < abs(le - mu(1));
    if ~any(lab) || all(lab), break; end
    mu = [mean(le(~lab)), mean(le(lab))];
end
if 10^(mu(2)-mu(1)) < 10
    warning('Разница уровень/пауза < 10 дБ — сегментация может быть ненадёжной');
end
act = medfilt1(double(env > 10^mean(mu)), 5) > 0.5;
dd  = diff([false; act; false]);
stE = find(dd ==  1);   fnE = find(dd == -1) - 1;
ok  = (fnE - stE + 1)*hopE >= round(0.4*fs);
stE = stE(ok);   fnE = fnE(ok);
nT  = numel(stE);
fprintf('\nНайдено активных сегментов: %d\n', nT);

%% 7. ИЗМЕРЕНИЯ НА КАЖДОМ СЕГМЕНТЕ ---------------------------------------
fT     = nan(nT,1);     isTone = false(nT,1);
AT     = nan(nT,N);     THD    = nan(nT,N);
dLT    = nan(nT,N-1);   phT    = nan(nT,N-1);
nH     = 5;             trimFrac = 0.10;

for s = 1:nT
    i1 = (stE(s)-1)*hopE + 1;
    i2 = min((fnE(s)-1)*hopE + winE, L);
    n  = i2 - i1 + 1;
    if n < round(0.4*fs), continue; end
    seg = data(i1:i2,:);

    fq = nan(1,4);                        % стабильность частоты по четвертям
    for q = 1:4
        a = i1 + round((q-1)*n/4);
        b = i1 + round(q*n/4) - 1;
        fq(q) = fft_peak_freq(seg(a-i1+1:b-i1+1, refCh), fs, fMaxAnal);
    end
    if any(isnan(fq)), continue; end
    isTone(s) = (max(fq) - min(fq)) < 5;  % иначе — свип
    if ~isTone(s), continue; end

    f0 = fft_peak_freq(seg(:,refCh), fs, fMaxAnal);
    if isnan(f0), continue; end
    [f0, phiR, okR] = demod_freq_phase(seg(:,refCh), f0, fs, trimFrac);
    if ~okR, continue; end
    fT(s) = f0;

    tt = (0:n-1)'/fs;   tc = tt - mean(tt);
    w  = hann(n);       sw = sum(w);
    e0 = exp(-1i*2*pi*f0*tc);
    for ch = 1:N
        yw = seg(:,ch) .* w;
        Ah = zeros(1,nH);
        for h = 1:nH
            if h*f0 >= 0.45*fs, break; end
            Ah(h) = 2*abs(sum(yw .* (e0.^h))) / sw;
        end
        AT(s,ch)  = Ah(1);
        THD(s,ch) = 100*sqrt(sum(Ah(2:end).^2)) / max(Ah(1), eps);
        if ch ~= refCh
            j = ch - (ch > refCh);
            dLT(s,j) = gccphat_lag(seg(:,refCh), seg(:,ch), fs, maxLag);
            [~, phiC, okC] = demod_freq_phase(seg(:,ch), f0, fs, trimFrac);
            if okC, phT(s,j) = phiC - phiR; end
        end
    end
end

iT = isTone & ~isnan(fT);
haveTones = sum(iT) >= 4;
if ~haveTones
    warning('Стабильных тонов не найдено (%d из %d) — анализ тонов и DOA пропущены', ...
        sum(iT), nT);
end
if haveTones
    fTv = fT(iT);  ATv = AT(iT,:);  THDv = THD(iT,:);
    dLTv = dLT(iT,:);  phTv = phT(iT,:);  nTv = numel(fTv);
else
    fTv = [];  ATv = [];  THDv = [];  dLTv = [];  phTv = [];  nTv = 0;
end
lag0Ms = nan(1, N-1);

%% 7.5 Фазовый МНК: задержка + полярность соседних пар ------------------
% Модель: dphi(f) = 2*pi*f*tau + pi*s; s = 0 (совпадение) или +-1
% (инверсия). Тоны ниже 0.98*fAliasAdj не заворачиваются, поэтому МНК
% по фазам работает напрямую и даёт полярность БЕЗ GCC.
fAliasAdj = cSnd/(2*dMic);                 % ~1039 Гц
tauAdjPh = nan(1,3);  polPh = nan(1,3);  resPh = nan(1,3);
tauRefPh = nan(1,3);  polRef = nan(1,3);  % отн. ch0 (накопление)
if haveTones
    phM = [wrap_pi(phTv(:,1)), wrap_pi(phTv(:,2)-phTv(:,1)), ...
           wrap_pi(phTv(:,3)-phTv(:,2))];
    phMask = fTv <= 0.98*fAliasAdj;
    cand = [-1 0 1];
    for j = 1:3
        okj = phMask & ~isnan(phM(:,j));
        if sum(okj) < 2, continue; end
        f_ = fTv(okj);   d_ = phM(okj,j);   sf2 = sum(f_.^2);
        Rall = nan(1,3);  tauAll = nan(1,3);
        for ci = 1:3
            sc = cand(ci);
            tauAll(ci) = sum(f_ .* (d_ - pi*sc)) / (2*pi*sf2);   % с
            rC = d_ - (2*pi*f_*tauAll(ci) + pi*sc);
            Rall(ci) = sqrt(mean(rC.^2));
        end
        [bestR, bi] = min(Rall);
        tauBest = tauAll(bi);   sBest = cand(bi);
        resPh(j) = bestR;
        R0 = Rall(2);  Rinv = min(Rall(1), Rall(3));
        if abs(R0 - Rinv) < 0.25
            polJ = NaN;                   % полярность неопределима
        elseif sBest == 0
            polJ = 1;
        else
            polJ = -1;
        end
        if abs(tauBest) > tauMaxMs/1000 || bestR > 0.5
            tauAdjPh(j) = NaN;  polPh(j) = NaN;   % недостоверно
        else
            tauAdjPh(j) = 1000*tauBest;  polPh(j) = polJ;   % мс
        end
    end
    for m = 1:3
        if ~any(isnan(tauAdjPh(1:m))), tauRefPh(m) = sum(tauAdjPh(1:m)); end
        if ~any(isnan(polPh(1:m))),    polRef(m)   = prod(polPh(1:m));   end
    end
    fprintf('\nФазовый МНК (соседние пары): задержки %s мкс; остатки %s рад\n', ...
        sprintf('%+7.1f ', tauAdjPh*1e3), sprintf('%.2f ', resPh));
end

%% ДИАГНОСТИКА ------------------------------------------------------------
fprintf('\n===== ДИАГНОСТИКА =====\n');
segDur = (fnE - stE + 1)*hopE/fs;
fprintf('Сегментов: %d; короче 0.4 с: %d\n', nT, ...
    sum((fnE-stE+1)*hopE < round(0.4*fs)));
fprintf('Длительности, с: медиана %.2f, min %.2f, max %.2f\n', ...
    median(segDur), min(segDur), max(segDur));
fprintf('Порог тишины: %.1f dBFS\n', 20*log10(10^mean(mu)));
fprintf('Тонов распознано: %d из %d\n', sum(iT), nT);
dF = nan(nT,1);  fA = nan(nT,1);
for s = 1:nT
    i1 = (stE(s)-1)*hopE+1;  i2 = min((fnE(s)-1)*hopE+winE, L);
    n = i2-i1+1;  if n < 64, continue; end
    fq = nan(1,4);
    for q = 1:4
        a = i1+round((q-1)*n/4);  b = i1+round(q*n/4)-1;
        fq(q) = fft_peak_freq(data(a:b,refCh), fs, fMaxAnal);
    end
    dF(s) = max(fq)-min(fq);  fA(s) = median(fq);
end
fprintf('Стабильных (<5 Гц разброс): %d, нестабильных: %d\n', ...
    sum(dF<5), sum(dF>=5));
fprintf('Пиковые частоты первых сегментов, Гц:\n');
fprintf('%.0f  ', round(fA(1:min(40,numel(fA)))));  fprintf('\n');
if sum(iT) > 0
    fprintf('Тонов ниже fLagMax=%g Гц: %d\n', fLagMax, sum(fT(iT)<=fLagMax));
    fprintf('phT NaN (пара 1): %d из %d\n', sum(isnan(phT(iT,1))), sum(iT));
end
fprintf('Не-тонов длиннее 3 с (свипы): %d\n', sum(~isTone & segDur>=3));

%% 8. ГРАФИКИ ПО ТОНАМ ----------------------------------------------------
if haveTones
    cols = lines(N-1);

    % 8.1 чувствительность
    figure('Name','Уровни и чувствительность по тонам');
    subplot(2,1,1);
    plot(fTv, 20*log10(ATv + eps), '.'); grid on;
    xlabel('Частота, Гц'); ylabel('Амплитуда 1-й гармоники, dBFS');
    title('Абсолютный уровень'); legend(compose('ch%d',0:N-1));
    subplot(2,1,2); hold on;
    hs = gobjects(1,N-1);
    for j = 1:N-1
        hs(j) = plot(fTv, 20*log10(ATv(:,j+1)./ATv(:,refCh) + eps), ...
                     '.', 'Color', cols(j,:));
    end
    grid on; xlabel('Частота, Гц'); ylabel('дБ отн. опорного'); hold off;
    title('Относительная чувствительность / АЧХ пар');
    legend(hs, compose('ch%d/ch0',1:N-1));

    % 8.2 задержка: точки — GCC (диагностика), линии — фазовый МНК
    figure('Name','Задержка по тонам','Color','w'); hold on;
    hs = gobjects(1,N-1);
    lowMask = fTv <= fLagMax;
    for j = 1:N-1
        hs(j) = plot(fTv, dLTv(:,j), '.', 'Color', cols(j,:));
        if any(lowMask)
            lag0Ms(j) = median(dLTv(lowMask,j), 'omitnan');   % диагностика
        end
        if ~isnan(tauRefPh(j))
            plot([fMinAnal fMaxAnal], tauRefPh(j)*[1 1], '-', ...
                 'Color', cols(j,:), 'LineWidth', 1.5);
        end
    end
    plot([fMinAnal fMaxAnal],  tauMaxMs*[1 1], 'k:');
    plot([fMinAnal fMaxAnal], -tauMaxMs*[1 1], 'k:');
    grid on; xlabel('Частота, Гц'); ylabel('Задержка, мс'); hold off;
    title('Точки — GCC (диагностика), линии — фазовый МНК, пунктир — предел d/c');
    legend(hs, compose('ch%d-ch0',1:N-1));

    % 8.3 полярность (фаза после компенсации МНК-задержкой)
    figure('Name','Полярность (фаза после компенсации)','Color','w');
    for j = 1:N-1
        subplot(N-1,1,j);
        if ~isnan(tauRefPh(j)) && any(~isnan(phTv(:,j)))
            phc = wrap_pi(phTv(:,j) - 2*pi*fTv*tauRefPh(j)/1000);
            plot(fTv, phc, '.'); hold on;
            plot([fMinAnal fMaxAnal], [0 0], 'g--');
            plot([fMinAnal fMaxAnal], [pi pi], 'r--');
            plot([fMinAnal fMaxAnal], [-pi -pi], 'r--'); hold off;
            if isnan(polRef(j)), verd = 'неопределена';
            elseif polRef(j) < 0, verd = 'ИНВЕРСИЯ';
            else, verd = 'совпадает'; end
            title(sprintf('ch%d-ch0: полярность — %s (фазовый МНК)', j, verd));
        else
            plot(fTv, phTv(:,j), '.');
            title(sprintf('ch%d-ch0: задержка/фаза не оценены', j));
        end
        grid on; ylim([-pi pi]); ylabel('рад');
    end
    xlabel('Частота, Гц');

    % 8.4 THD
    figure('Name','THD по тонам');
    plot(fTv, THDv); grid on; set(gca,'YScale','log');
    xlabel('Частота, Гц'); ylabel('THD, %');
    title('Нелинейные искажения (гармоники 2-5)');
    legend(compose('ch%d',0:N-1));

    % 8.5 линейность (наклон дБ/дБ, норма = 1)
    lv = [25 50 75 100];
    slopeAll = [];  slopeCh = [];  i = 1;
    while i <= nTv
        j2 = i;
        while j2 < nTv && abs(fTv(j2+1)-fTv(i)) < 1, j2 = j2+1; end
        if j2 - i + 1 == numel(lv)
            xdb = 20*log10(lv);
            for ch = 1:N
                ydb = sort(20*log10(ATv(i:j2,ch) + eps));
                p = polyfit(xdb, ydb, 1);
                slopeAll(end+1,1) = p(1);   %#ok<AGROW>
                slopeCh(end+1,1)  = ch;     %#ok<AGROW>
            end
        end
        i = j2 + 1;
    end
    if ~isempty(slopeAll)
        figure('Name','Линейность тракта (наклон дБ/дБ)');
        for ch = 1:N
            subplot(2,2,ch);
            histogram(slopeAll(slopeCh==ch), 0.8:0.01:1.2); grid on;
            title(sprintf('ch%d (медиана %.3f)', ch-1, ...
                median(slopeAll(slopeCh==ch))));
            xlabel('наклон, дБ/дБ');
        end
    end
end

fprintf('\nКлиппинг: сэмплов выше 0.995 FS:\n');
for ch = 1:N
    fprintf('  ch%d: %.4f %%\n', ch-1, 100*sum(abs(data(:,ch))>=0.995)/L);
end

%% 9. ШУМ В ПАУЗАХ ---------------------------------------------------------
noiseMask = false(L,1);
for s = 1:numel(stE)
    a = (fnE(s)-1)*hopE + winE + round(0.3*fs);
    if s < numel(stE)
        b = (stE(s+1)-1)*hopE - round(0.05*fs);
    else
        b = L;
    end
    a = max(a,1);  b = min(b,L);
    if b > a, noiseMask(a:b) = true; end
end
noiseRms = nan(1, N);
if nnz(noiseMask) >= round(2*fs)
    figure('Name','Спектр шума (паузы)'); hold on;
    for ch = 1:N
        [Pn, Fn] = pwelch(data(noiseMask,ch), hann(round(0.25*fs)), ...
                          round(0.125*fs), 2^nextpow2(round(0.25*fs)), fs);
        plot(Fn, 10*log10(Pn + eps));
    end
    grid on; xlim([0 min(5000, fs/2)]);
    xlabel('Частота, Гц'); ylabel('PSD, дБ/Гц');
    title('Шумовой пол в паузах'); legend(compose('ch%d',0:N-1)); hold off;
    noiseRms = sqrt(mean(data(noiseMask,:).^2, 1));
else
    warning('Паузы не найдены — оценка шума пропущена');
end

%% 10-11. DOA ЛИНЕЙНОЙ РЕШЁТКИ, БЛИЖНЕЕ ПОЛЕ, ДН, СОХРАНЕНИЕ --------------
% Угол: 0° = по нормали к центру; +90° = сторона +X (ch3).
% GCC по одиночным тонам в оценках НЕ используется (методически непригоден).

trueAngle = NaN;
mAng = regexp(pwd, 'Угол_?(-?\d+)', 'once', 'tokens');
if isempty(mAng)
    mAng = regexp(pwd, '(?i)angle_?(-?\d+)', 'once', 'tokens');
end
if ~isempty(mAng)
    trueAngle = str2double(mAng{1});
    fprintf('\nЗаданный угол (из имени папки): %g°\n', trueAngle);
end

angleDeg = NaN;  srcPos = [NaN NaN];  u = [NaN NaN];
tauAdjGCC = nan(1,3);  tauSweepMs = nan(1,N-1);  swAdjMs = nan(1,3);
thetaFull = nan(nT,1);  angleFromNF = false;

%% --- 10.2 Широкополосные задержки по СВИПАМ (GCC-PHAT) ---
swTau = [];
for s = 1:nT
    if isTone(s), continue; end
    i1 = (stE(s)-1)*hopE + 1;
    i2 = min((fnE(s)-1)*hopE + winE, L);
    if (i2 - i1 + 1) < round(3.0*fs), continue; end
    cLen = min(i2 - i1 + 1, round(2.0*fs));
    a = i1 + floor((i2 - i1 + 1 - cLen)/2);
    b = a + cLen - 1;
    x = data(a:b, refCh);
    row = nan(1, N-1);
    for ch = 1:N
        if ch == refCh, continue; end
        j = ch - (ch > refCh);
        row(j) = gccphat_lag(x, data(a:b,ch), fs, maxLag);
    end
    swTau(end+1,:) = row;   %#ok<AGROW>
end
if ~isempty(swTau)
    tauSweepMs = median(swTau, 1, 'omitnan');
    swAdjMs = [tauSweepMs(1), tauSweepMs(2)-tauSweepMs(1), ...
               tauSweepMs(3)-tauSweepMs(2)];
    fprintf('Задержки по свипам (отн. ch0), мс: %s\n', sprintf('%+8.4f ', tauSweepMs));
    if any(abs(swAdjMs) > tauMaxMs)
        fprintf(['  ВНИМАНИЕ: задержки свипов превышают d/c — межканальные ', ...
                 'смещения (сведение/электроника)? В оценках не используются.\n']);
    end
end

%% --- 10.3 Диагностика GCC по тонам и график DOA по тонам ---
if haveTones
    lowMask = fTv <= fLagMax;
    if any(lowMask)
        tauAdjGCC = median([dLTv(lowMask,1), dLTv(lowMask,2)-dLTv(lowMask,1), ...
                            dLTv(lowMask,3)-dLTv(lowMask,2)], 1, 'omitnan');
    end

    phM = [wrap_pi(phTv(:,1)), wrap_pi(phTv(:,2)-phTv(:,1)), ...
           wrap_pi(phTv(:,3)-phTv(:,2))];
    phMask = fTv <= 0.98*fAliasAdj;
    thT = nan(nTv, 3);
    for j = 1:3
        okj = phMask & ~isnan(phM(:,j));
        if any(okj)
            sT = cSnd*(phM(:,j)./(2*pi*fTv))/dMic;
            thT(:,j) = asind(min(max(sT,-1),1));
        end
    end
    thT(~phMask,:) = NaN;
    thetaFull(iT) = median(thT, 2, 'omitnan');

    figure('Name','DOA по тонам (соседние пары)','Color','w');
    plot(fTv, thT, '.'); grid on; hold on;
    cols3 = lines(3);
    leg = arrayfun(@(a,b) sprintf('пара ch%d-ch%d', a, b), ...
                   [0 1 2], [1 2 3], 'UniformOutput', false);
    for j = 1:3
        if ~isnan(tauAdjPh(j))
            thPhj = asind(min(max(cSnd*tauAdjPh(j)/1000/dMic,-1),1));
            plot([100 2000], thPhj*[1 1], '-', 'Color', cols3(j,:), 'LineWidth', 1.5);
            leg{end+1} = sprintf('МНК, пара %d', j);
        end
    end
    if ~isnan(trueAngle)
        plot([100 2000], trueAngle*[1 1], 'r--', 'LineWidth', 1.5);
        leg{end+1} = 'задано';
    end
    hold off; xlim([100 2000]); ylim([-90 90]);
    xlabel('Частота тона, Гц'); ylabel('Угол, °');
    title(sprintf(['Точки — по тонам (фаза соседних пар); линии — МНК; ', ...
        'f > %d Гц скрыты (неоднозначность)'], round(0.98*fAliasAdj)));
    legend(leg);
end

%% --- 10.4 Дальнее поле: итоговый угол (БЕЗ GCC по тонам) ---
est = [];  estName = {};
for j = 1:3
    if ~isnan(tauAdjPh(j))
        est(end+1,1) = asind(min(max(cSnd*tauAdjPh(j)/1000/dMic,-1),1)); %#ok<AGROW>
        estName{end+1} = sprintf('фаза-МНК, пара %d', j);                %#ok<AGROW>
    end
end
if ~any(isnan(swAdjMs))
    for j = 1:3
        if abs(swAdjMs(j)) <= tauMaxMs
            est(end+1,1) = asind(min(max(cSnd*swAdjMs(j)/1000/dMic,-1),1)); %#ok<AGROW>
            estName{end+1} = sprintf('свип, пара %d', j);                 %#ok<AGROW>
        end
    end
end
if ~isempty(est)
    angleDeg = median(est);
    u = [sind(angleDeg), cosd(angleDeg)];
end

%% --- 10.5 Ближнее поле: положение источника (с невязкой) ---
srcTau = [];  srcTauName = '';
if ~any(isnan(swAdjMs)) && all(abs(swAdjMs) <= tauMaxMs)
    srcTau = swAdjMs;    srcTauName = 'свипы';
elseif sum(~isnan(tauAdjPh)) >= 2
    srcTau = tauAdjPh;   srcTauName = 'фаза-МНК тонов';
end
if ~isempty(srcTau)
    vIdx = find(~isnan(srcTau) & abs(srcTau) <= tauMaxMs);
    tauAdjSec = srcTau(vIdx)/1000;          % с
    pairLo = vIdx(:);   pairHi = vIdx(:) + 1;
else
    tauAdjSec = [];
end
if ~isempty(tauAdjSec) && max(abs(tauAdjSec)) > 15e-6
    tauCol = tauAdjSec(:);
    costNF = @(sv) sum(((hypot(micPos(pairLo,1)-sv(1), micPos(pairLo,2)-sv(2)) ...
                       - hypot(micPos(pairHi,1)-sv(1), micPos(pairHi,2)-sv(2)))/cSnd ...
                       - tauCol).^2);
    xg = -1.5:0.05:1.5;   yg = 0.1:0.05:6;   % диапазон поиска — подстройте под дистанцию
    [Xg, Yg] = meshgrid(xg, yg);
    Cg = arrayfun(@(xx,yy) costNF([xx yy]), Xg, Yg);
    [~, km] = min(Cg(:));
    sf = fminsearch(costNF, [Xg(km), Yg(km)], optimset('Display','off'));
    if all(isfinite(sf))
        sf(2) = abs(sf(2));                  % источник спереди (y > 0)
        srcPos = sf;
        mTau = (hypot(micPos(pairLo,1)-sf(1), micPos(pairLo,2)-sf(2)) ...
              - hypot(micPos(pairHi,1)-sf(1), micPos(pairHi,2)-sf(2)))/cSnd;
        resNF = sqrt(mean((mTau - tauCol).^2));
        fprintf('\nБлижнее поле (по %s): x=%+.3f м, y=%.3f м, R=%.3f м, угол %+.2f°, невязка %.0f мкс\n', ...
            srcTauName, srcPos(1), srcPos(2), hypot(srcPos(1),srcPos(2)), ...
            atan2d(srcPos(1),srcPos(2)), resNF*1e6);
        if resNF > 30e-6
            fprintf(['  ВНИМАНИЕ: невязка велика — данные не соответствуют одному ', ...
                     'точечному источнику (отражения? фазовые сдвиги каналов?). ', ...
                     'Положение недостоверно.\n']);
        else
            angleDeg = atan2d(srcPos(1), srcPos(2));
            angleFromNF = true;
            u = [sind(angleDeg), cosd(angleDeg)];
        end
        if abs(atan2d(srcPos(1),srcPos(2))) < 5
            fprintf('(при углах ~0° дистанция по TDOA определяется плохо)\n');
        end
    end
elseif ~isempty(tauAdjSec)
    fprintf('\nВсе задержки ~0: источник по центру (~0°), дистанция по TDOA не определяется\n');
end

%% --- 10.6 ДН решётки (Bartlett, по измеренным тонам) ---
if haveTones
    fBeam = [200 500 1000 1900];
    thScan = -90:0.25:90;
    xMic = micPos(:,1);
    figure('Name','ДН решётки (Bartlett)','Color','w');
    for bi = 1:numel(fBeam)
        iBAll = find(abs(fTv - fBeam(bi)) < 1);
        if isempty(iBAll), continue; end
        [~, iBest] = max(ATv(iBAll, refCh));   % самый громкой повтор (100 %)
        iB = iBAll(iBest);
        f0 = fTv(iB);
        z = ATv(iB,:) .* exp(1i*[0, phTv(iB,:)]);   % комплексные амплитуды
        if any(isnan(z)), continue; end
        Bp = zeros(size(thScan));
        for ti = 1:numel(thScan)
            Bp(ti) = abs(z * exp(-1i*2*pi*f0*xMic*sind(thScan(ti))/cSnd))^2;
        end
        Bp = 10*log10(Bp/max(Bp) + eps);
        subplot(2,2,bi);
        plot(thScan, Bp, 'LineWidth', 1); grid on; hold on;
        yl = [-25 3];
        if ~isnan(angleDeg),  plot(angleDeg*[1 1],  yl, 'r--'); end
        if ~isnan(trueAngle), plot(trueAngle*[1 1], yl, 'm:');  end
        hold off; xlim([-90 90]); ylim(yl);
        xlabel('Угол сканирования, °'); ylabel('дБ');
        if f0 > fAliasAdj
            title(sprintf('%d Гц — выше f_гр: дифракционные лепестки', round(f0)));
        else
            title(sprintf('%d Гц', round(f0)));
        end
    end
end

%% --- 10.7 Сводка DOA и сохранение ---
fprintf('\n--- СВОДКА DOA (линейка, d=%d мм, f_гр=%d Гц) ---\n', ...
    round(1000*dMic), round(fAliasAdj));
fprintf('Соседние задержки, мкс (tau>0 — старший канал раньше):\n');
fprintf('%10s %16s %12s %10s %12s\n', 'пара', 'GCC-тоны(диагн)', 'фаза-МНК', 'остаток', 'свип');
for j = 1:3
    fprintf('  ch%d-ch%d %14.1f %12.1f %10.2f %12.1f\n', j-1, j, ...
        tauAdjGCC(j)*1e3, tauAdjPh(j)*1e3, resPh(j), swAdjMs(j)*1e3);
end
fprintf(['  (!) GCC по одиночным тонам непригоден (нет полосы) и в оценках ', ...
         'не используется; > %.0f мкс (d/c) физически невозможны.\n'], tauMaxMs*1e3);
if any(~isnan(polPh))
    fprintf('Полярность соседних пар (фазовый МНК):');
    for j = 1:3
        if isnan(polPh(j)), p = '?';
        elseif polPh(j) < 0, p = 'ИНВЕРСИЯ';
        else, p = 'совпадает'; end
        fprintf('  ch%d-ch%d: %s', j-1, j, p);
    end
    fprintf('\n');
end
if ~isempty(est)
    fprintf('Оценки угла (дальнее поле):\n');
    for e = 1:numel(est)
        fprintf('  %-22s %+7.2f°\n', estName{e}, est(e));
    end
    if numel(est) > 1 && (max(est) - min(est)) > 10
        fprintf('  ВНИМАНИЕ: оценки расходятся на %.1f° — итог ненадёжен\n', ...
            max(est) - min(est));
    end
    if angleFromNF
        fprintf('  ИТОГ (ближнее поле)     %+7.2f°\n', angleDeg);
    else
        fprintf('  ИТОГ (медиана)          %+7.2f°\n', angleDeg);
    end
    if ~isnan(trueAngle)
        fprintf('  Задано %g°, ошибка %+.2f°\n', trueAngle, ...
            wrap_180(angleDeg - trueAngle));
    end
else
    fprintf('Оценок угла нет: нет валидных задержек (см. причины выше)\n');
end

if saveResults
    Ttab = table(fT, isTone, AT, THD, dLT, phT, thetaFull, ...
        'VariableNames', {'f_Hz','isTone','Amp','THD_pct','delay_ms','phase_rad','angle_deg'});
    writetable(Ttab, 'tones.csv');
    save('analysis_results.mat', 'fs', 'micPos', 'fT', 'isTone', 'AT', 'THD', ...
         'dLT', 'phT', 'noiseRms', 'lag0Ms', 'tauAdjPh', 'polPh', 'resPh', ...
         'tauRefPh', 'polRef', 'tauSweepMs', 'swAdjMs', 'angleDeg', 'srcPos', ...
         'trueAngle', 'u', 'thetaFull');
    fprintf('Сохранено: tones.csv, analysis_results.mat\n');
end

%% 12. СВОДКА ---------------------------------------------------------------
fprintf('\n==================== СВОДКА ====================\n');
fprintf('Каналов: %d, fs = %g Гц, длительность %.1f с\n', N, fs, L/fs);
if haveTones
    fprintf('\nЧувствительность (медиана, дБ отн. ch%d) и THD:\n', refCh-1);
    aRef = max(median(ATv(:,refCh),'omitnan'), eps);
    for ch = 1:N
        fprintf('  ch%d: %+6.2f дБ   THD(медиана) %.2f %%\n', ch-1, ...
            20*log10(median(ATv(:,ch),'omitnan')/aRef), ...
            median(THDv(:,ch),'omitnan'));
    end
    fprintf('\nЗадержки и полярность отн. ch0 (фазовый МНК по тонам):\n');
    for m = 1:N-1
        if isnan(tauRefPh(m))
            fprintf('  ch%d-ch0: не оценена\n', m);
        else
            if isnan(polRef(m)), verd = 'неопределена';
            elseif polRef(m) < 0, verd = 'ИНВЕРСИЯ';
            else, verd = 'совпадает'; end
            fprintf('  ch%d-ch0: %+7.1f мкс   полярность: %s\n', ...
                m, tauRefPh(m)*1e3, verd);
        end
    end
    if ~any(isnan(noiseRms))
        fprintf('\nШум и SNR:\n');
        sigRms = median(ATv, 1, 'omitnan') / sqrt(2);
        for ch = 1:N
            fprintf('  ch%d: шум %6.1f dBFS, сигнал %6.1f dBFS, SNR %5.1f дБ\n', ...
                ch-1, 20*log10(max(noiseRms(ch),eps)), ...
                20*log10(max(sigRms(ch),eps)), ...
                20*log10(max(sigRms(ch),eps)/max(noiseRms(ch),eps)));
        end
        if 20*log10(max(median(sigRms),eps)) < -40
            fprintf(['  Совет: уровень сигнала очень низкий при нулевом клиппинге ', ...
                     '— поднимите уровень воспроизведения на 20-30 дБ.\n']);
        end
    end
    if ~isnan(angleDeg)
        fprintf('\nНаправление на источник: %.1f° (0° = по центру, +90° = сторона ch3)\n', ...
            angleDeg);
    end
end
fprintf('=================================================\n');

%% ======================= ЛОКАЛЬНЫЕ ФУНКЦИИ ==============================

function [lagMs, pol] = gccphat_lag(x, y, fs, maxLag)
% GCC-PHAT + параболическая интерполяция пика.
% ВНИМАНИЕ: на узкополосных сигналах (одиночный тон) НЕРАБОТОСПОСОБЕН.
% lagMs > 0 => y приходит РАНЬШЕ x.
nfft = 2^nextpow2(numel(x));
G = fft(x, nfft) .* conj(fft(y, nfft));
r = real(ifft(G ./ (abs(G) + 1e-20)));
maxLag = max(1, min(maxLag, floor(nfft/2)-1));
c = [r(end-maxLag+1:end); r(1:maxLag+1)];
[~, im] = max(abs(c));
pol = sign(c(im));
lag = im - (maxLag + 1);
if im > 1 && im < numel(c)
    a = c(im-1);  b = c(im);  cc = c(im+1);
    den = a - 2*b + cc;
    if abs(den) > 1e-12
        lag = lag + 0.5*(a - cc)/den;
    end
end
lagMs = lag / fs * 1000;
end

function f = fft_peak_freq(x, fs, fmax, fmin)
% Частота главного пика: окно Ханна, паддинг x4, парабола по log|X|.
% fmin (по умолчанию 70 Гц) отсекает сетевой фон 50 Гц.
if nargin < 4, fmin = 70; end
x = x(:);
n = numel(x);
if n < 16, f = NaN; return; end
nfft = 2^nextpow2(4*n);
X = abs(fft(x .* hann(n), nfft));
lo = max(2, round(fmin/fs*nfft));
hi = max(lo+2, min(round(fmax/fs*nfft), floor(nfft/2)));
[~, ip] = max(X(lo:hi));   ip = ip + lo - 1;
if ip <= 1 || ip >= numel(X), f = NaN; return; end
a = log(X(ip-1)+eps);  b = log(X(ip)+eps);  c = log(X(ip+1)+eps);
den = a - 2*b + c;
if abs(den) < 1e-12
    f = (ip-1)*fs/nfft;
else
    f = (ip - 1 + 0.5*(a-c)/den) * fs/nfft;
end
end

function [f, phi, ok] = demod_freq_phase(x, f0, fs, trimFrac)
% Уточнение частоты тона и его фазы в ЦЕНТРЕ сегмента.
% Сигнал перемножается с exp(-i*2*pi*f0*tc) на ОБЩЕЙ шкале времени
% (центр сегмента = 0). Фазы в центрах 8 окон дают d = f_ист - f0
% (однозначно при |d| < ~1.2 Гц), их разброс — критерий стабильности.
x = x(:);
n = numel(x);
i1 = round(trimFrac*n) + 1;
i2 = n - round(trimFrac*n);
m = i2 - i1 + 1;
if m < 64, f = f0; phi = NaN; ok = false; return; end
xc = x(i1:i2);
tc = ((0:m-1)')/fs;
tc = tc - mean(tc);

w  = hann(m);
X0 = sum(xc .* w .* exp(-1i*2*pi*f0*tc));
phi = angle(X0);

Nc = 8;
q  = floor(m/Nc);
if q < 32, f = f0; ok = false; return; end
pq = nan(1,Nc);
for kq = 1:Nc
    a = (kq-1)*q + 1;
    b = min(kq*q, m);
    pq(kq) = angle(sum(xc(a:b) .* hann(b-a+1) .* ...
                        exp(-1i*2*pi*f0*tc(a:b))));
end
dt  = q/fs;
dp  = wrap_pi(diff(pq));
df  = mean(dp)/(2*pi*dt);
f   = f0 + df;
dev = wrap_pi(dp - 2*pi*df*dt);
ok  = ~any(isnan(dev)) && max(abs(dev)) < 0.6;
end

function a = wrap_pi(x)
a = mod(x + pi, 2*pi) - pi;
end

function a = wrap_180(x)
a = mod(x + 180, 360) - 180;
end