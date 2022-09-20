% Tutorial_7_2.m
%
% This code produces those dynamical regimes possible with two units,
% with a firing-rate model of a neural circuit.
% Two units (representing neural
% assemblies) are simulated with firing rates that increase linearly with
% input current above a system-dependent threshold (Ithresh) and that saturate at a value
% rmax = 100Hz.
% Units can excite or inhibit each other to different degrees according to
% the connectivity matrix W in which W(i,j) represents connection strength
% from unit j to unit i.
% The flag "attractor-flag" is used to determine the type of simulation.
%
% This code is a solution of Tutorial 7.2 in the textbook,
% An Introductory Course in Computational Neuroscience
% by Paul Miller, Brandeis University, 2017
%%

clear

tmax = 2;  % default value of maximum time to simulate

% case 7 produces two distinct oscillators of different
% frequencies. Changes in initial conditions -- or a current pulse
% -- can switch between oscillators.
Ithresh = [-15; -12; 1; 2];
W = [1.2 -0.5 0.5 0; -1.5 1.9 0 1.2; -1.1 -0.6 0 -0.8; -1.5 -1.3 -0.8 0];
rinit1 = [0; 15; 0; 5];

Ntrials = 50;
Nunits = 4;

rng(1)

%% Figure positions for subplots
set(0,'DefaultLineLineWidth',2,...
    'DefaultLineMarkerSize',8, ...
    'DefaultAxesLineWidth',2, ...
    'DefaultAxesFontSize',14,...
    'DefaultAxesFontWeight','Bold');
figure(1)
clf

%% Set up the time vector
dt = 0.001;
tvec = 0:dt:tmax;
Nt = length(tvec);

r = zeros(Nunits,Nt);   % array of rate of each cell as a function of time
rmax = 100;             % maximum firing rate
tau = 0.010;            % base time constant for changes of rate

sigma_noise = 0.6/sqrt(dt);

all_r = zeros(Ntrials, Nunits, length(tvec));

%% Set up axes and label them for the figures

for trial = 1:Ntrials
    Inoise = sigma_noise*randn(size(r));
    %     if ( trial == 1 )
    r(:,1) = rinit1;                % Initialize firing rate
    %     else
    %         r(:,1) = rinit2;                % Initialize firing rate
    %     end
    %% Now simulate through time
    for i=2:Nt
        I = W'*r(:,i-1) + Inoise(:,i-1);                   % total current to each unit
        newr = r(:,i-1) + dt/tau*(I-Ithresh-r(:,i-1));  % Euler-Mayamara update of rates
        r(:,i) = max(newr,0);                           % rates are not negative
        r(:,i) = min(r(:,i),rmax);                      % rates can not be above rmax
        
    end

    all_r(trial,:,:)=r;
    
    
    subplot(Ntrials/2,2,trial)
    plot(tvec,r(1,:),'m')
    hold on
    plot(tvec,r(4,:),'b:')
    plot(tvec,r(2,:),'r')
    plot(tvec,r(3,:),'c:')
    axis([0 1.5 0 75])

    
end

%%

temp_inds = [1 3 7 8 10 11 14 17 18 23 25 27 28 29 30 31 32 37 40 42 43 44 45 46 47 48 49];
%                   1   2   3   4   5     6    7    8   9   10  11  12   
temp_transition_inds = [500 500 1100 238 1600 730 1000 477 950 1100 ... 
                   1300 320 350 400 360 1300 360  1200 725 500 ...
                   1050 820 260 830 210 800 350];
inds = temp_inds(temp_transition_inds > 350);
transition_inds = temp_transition_inds(temp_transition_inds > 350);

selected_r = all_r(inds,:,:);
sel_u1 = squeeze(sum(selected_r(:,1:2,:),2));
sel_u2 = squeeze(sum(selected_r(:,3:4,:),2));

%bandpass_u1 = bandpass(sel_u1', [10,20], 1/dt)';
%bandpass_u2 = bandpass(u2_lfp, [10,20], 1/dt);

bandpass_u1 = zeros(size(sel_u1));
bandpass_u2 = zeros(size(sel_u2));

hilbert_u1 = zeros(size(sel_u1));
hilbert_u2 = zeros(size(sel_u2));

phase_u1 = zeros(size(sel_u1));
phase_u2 = zeros(size(sel_u2));

for i = 1:length(inds)
    bandpass_u1(i,:) = bandpass(sel_u1(i,:), [10,40], 1/dt);
    bandpass_u2(i,:) = bandpass(sel_u2(i,:), [10,40], 1/dt);

    hilbert_u1(i,:) = hilbert(bandpass_u1(i,:));
    hilbert_u2(i,:) = hilbert(bandpass_u2(i,:));

    %hilbert_u1(i,:) = hilbert(sel_u1(i,:));
    %hilbert_u2(i,:) = hilbert(sel_u2(i,:));

    phase_u1(i,:) = angle(hilbert_u1(i,:));
    phase_u2(i,:) = angle(hilbert_u2(i,:));
end


phase_diff_raw = phase_u2 - phase_u1;
phase_diff = exp(-1j * phase_diff_raw);

for i = 1:length(inds)
    subplot(ceil(length(inds)/2),2,i)
    plot(sel_u1(i,:)')
    hold on
    plot(sel_u2(i,:)')
    title(i)
end


% Pull out snippets of data and transitions
window = 350;
snip_dat = zeros(2,length(inds), window*2);
band_snip_dat = zeros(2,length(inds), window*2);
snip_phase = zeros(2,length(inds), window*2);
snip_phase_diff = zeros(length(inds), window*2);

for i = 1:length(inds)
    snip_dat(1,i,:) = sel_u1(i,(transition_inds(i)-window):(transition_inds(i)+window-1));
    snip_dat(2,i,:) = sel_u2(i,(transition_inds(i)-window):(transition_inds(i)+window-1));
    band_snip_dat(1,i,:) = bandpass_u1(i,(transition_inds(i)-window):(transition_inds(i)+window-1));
    band_snip_dat(2,i,:) = bandpass_u2(i,(transition_inds(i)-window):(transition_inds(i)+window-1));
    snip_phase(1,i,:) = phase_u1(i,(transition_inds(i)-window):(transition_inds(i)+window-1));
    snip_phase(2,i,:) = phase_u2(i,(transition_inds(i)-window):(transition_inds(i)+window-1));
    snip_phase_diff(i,:) = phase_diff(i,(transition_inds(i)-window):(transition_inds(i)+window-1));
end

cols = 4;
for i = 1:length(inds)
    subplot(length(inds),cols,(cols*i)-3)
    plot(squeeze(snip_dat(1,i,:)))
    hold on
    plot(squeeze(snip_dat(2,i,:)))
    subplot(length(inds),cols,(cols*i)-2)
    plot(squeeze(band_snip_dat(1,i,:)))
    hold on
    plot(squeeze(band_snip_dat(2,i,:)))
    subplot(length(inds),cols,(cols*i)-1)
    plot(squeeze(snip_phase(1,i,:)))
    hold on
    plot(squeeze(snip_phase(2,i,:)))
    subplot(length(inds),cols,(cols*i))
    plot(angle(snip_phase_diff(i,:)))
end

%phase_coh = abs(mean(snip_phase_diff,1));
%plot(phase_coh)

%%
% Bootstrap phase coherence
full_inds = 1:length(inds);
samples = 100;
boot_phase_diff = zeros(samples, size(snip_phase_diff,1), size(snip_phase_diff,2));
boot_phase= zeros(samples, size(snip_phase,1), size(snip_phase,2), size(snip_phase,3));

for i = 1:samples
    this_inds = randsample(full_inds, length(full_inds), true);
    boot_phase_diff(i,:,:) = snip_phase_diff(this_inds,:);
    boot_phase(i,:,:,:) = snip_phase(:,this_inds,:);
end

boot_phase_coh = squeeze(abs(mean(boot_phase_diff,2)));

med_phase_coh = median(boot_phase_coh,1);
std_phase_coh = std(boot_phase_coh,1);

down_rate = 10;
med_phase_coh = med_phase_coh(1:down_rate:end);
std_phase_coh = std_phase_coh(1:down_rate:end);

errorbar(med_phase_coh, std_phase_coh)
hold on
plot(med_phase_coh)
xline(window/down_rate)
xlabel('Time (10s of ms)')
ylabel('Coherence (0-1)')

%% Save outputs for analysis in python
save('boot_phase_diff', "boot_phase_diff")
save('selected_trials','selected_r')
save('boot_phase','boot_phase')
