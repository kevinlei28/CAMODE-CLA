%% ============ IMODE Case Study Execution Script ============
% Purpose: Run a single instance of IMODE on CEC2020 F10 
% to capture evolutionary dynamics for Case Study analysis.
% ============================================================
clc;
clear all;
format short e;

%% 1. Experiment Settings
I_fno = 8;          % Target Problem: CEC2020 F10
Run_ID = 1;          % Seed / Run ID
Par = Introd_Par(I_fno); 

fprintf('==============================================\n');
fprintf('Starting Case Study Run\n');
fprintf('Problem: CEC2020 F%d\n', I_fno);
fprintf('Dimension: %dD\n', Par.n);
fprintf('Max FES: %d\n', Par.Max_FES);
fprintf('==============================================\n');

%% 2. Run IMODE with Logging
% Call the logging function directly. 
% Ensure IMODE_main_logging.m is in the same folder.
t_start = tic;

[outcome, com_time, SR, Avg_FES, res_det, bestx, CaseStudyData] = ...
    IMODE_main_logging(Run_ID, I_fno);

t_end = toc(t_start);

%% 3. Display Results
fprintf('\nRun Completed.\n');
fprintf('Computational Time: %.4f seconds\n', t_end);
fprintf('Best Fitness Error: %.4e\n', outcome);
fprintf('Success Rate (SR): %d\n', SR);

%% 4. Save Data for Plotting
% Define save path
base_path = '.\record\CaseStudy\';
if ~exist(base_path, 'dir')
    mkdir(base_path);
end

% File name: CaseStudy_F10_Run1.mat
save_filename = sprintf('CaseStudy_F%d_Run%d.mat', I_fno, Run_ID);
full_save_path = fullfile(base_path, save_filename);

save(full_save_path, 'CaseStudyData', 'outcome', 'bestx', 'Par');

fprintf('Data successfully saved to:\n%s\n', full_save_path);
fprintf('You can now run the plotting script to generate the Case Study figures.\n');