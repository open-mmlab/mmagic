function NIQE = calc_scores(input_image_path,shave_width)

%% Loading model
load modelparameters.mat
blocksizerow    = 96;
blocksizecol    = 96;
blockrowoverlap = 0;
blockcoloverlap = 0;
%% Calculating scores
NIQE = [];
    input_image = convert_shave_image(imread(input_image_path),shave_width);
        
    % Calculating scores
    NIQE = computequality(input_image,blocksizerow,blocksizecol,...
        blockrowoverlap,blockcoloverlap,mu_prisparam,cov_prisparam);

end


