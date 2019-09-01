function  [mu_prisparam cov_prisparam]  = estimatemodelparam(folderpath,...
    blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap,sh_th)
    
% Input
% folderpath      - Folder containing the pristine images
% blocksizerow    - Height of the blocks in to which image is divided
% blocksizecol    - Width of the blocks in to which image is divided
% blockrowoverlap - Amount of vertical overlap between blocks
% blockcoloverlap - Amount of horizontal overlap between blocks
% sh_th           - The sharpness threshold level
%Output
%mu_prisparam  - mean of multivariate Gaussian model
%cov_prisparam - covariance of multivariate Gaussian model

% Example call

%[mu_prisparam cov_prisparam] = estimatemodelparam('pristine',96,96,0,0,0.75);


%----------------------------------------------------------------
% Find the names of images in the folder
current = pwd;
cd(sprintf('%s',folderpath))
names        = ls;
names        = names(3:end,:);
cd(current)
% ---------------------------------------------------------------
%Number of features
% 18 features at each scale
featnum      = 18;
% ---------------------------------------------------------------
% Make the directory for storing the features
mkdir(sprintf('local_risquee_prisfeatures'))
% ---------------------------------------------------------------
% Compute pristine image features
for itr = 1:size(names,1)
itr
im               = imread(sprintf('%s\\%s',folderpath,names(itr,:)));
if(size(im,3)==3)
im               = rgb2gray(im);
end
im               = double(im);             
[row col]        = size(im);
block_rownum     = floor(row/blocksizerow);
block_colnum     = floor(col/blocksizecol);
im               = im(1:block_rownum*blocksizerow, ...
                   1:block_colnum*blocksizecol);               
window           = fspecial('gaussian',7,7/6);
window           = window/sum(sum(window));
scalenum         = 2;
warning('off')

feat = [];


for itr_scale = 1:scalenum

    
mu                       = imfilter(im,window,'replicate');
mu_sq                    = mu.*mu;
sigma                    = sqrt(abs(imfilter(im.*im,window,'replicate') - mu_sq));
structdis                = (im-mu)./(sigma+1);
              
               
               
feat_scale               = blkproc(structdis,[blocksizerow/itr_scale blocksizecol/itr_scale], ...
                           [blockrowoverlap/itr_scale blockcoloverlap/itr_scale], ...
                           @computefeature);
feat_scale               = reshape(feat_scale,[featnum ....
                           size(feat_scale,1)*size(feat_scale,2)/featnum]);
feat_scale               = feat_scale';


if(itr_scale == 1)
sharpness                = blkproc(sigma,[blocksizerow blocksizecol], ...
                           [blockrowoverlap blockcoloverlap],@computemean);
sharpness                = sharpness(:);
end


feat                     = [feat feat_scale];

im =imresize(im,0.5);

end

save(sprintf('local_risquee_prisfeatures\\prisfeatures_local%d.mat',...
    itr),'feat','sharpness');
end



%----------------------------------------------
% Load pristine image features
prisparam    = [];
current      = pwd;
cd(sprintf('%s','local_risquee_prisfeatures'))
names        = ls;
names        = names(3:end,:);
cd(current)
for itr      = 1:size(names,1)
% Load the features and select the only features     
load(sprintf('local_risquee_prisfeatures\\%s',strtrim(names(itr,:))));
IX               = find(sharpness(:) >sh_th*max(sharpness(:)));
feat             = feat(IX,:);
prisparam        = [prisparam; feat];

end 
%----------------------------------------------
% Compute model parameters
mu_prisparam       = nanmean(prisparam);
cov_prisparam      = nancov(prisparam);
%----------------------------------------------
% Save features in the mat file
save('modelparameters_new.mat','mu_prisparam','cov_prisparam');
%----------------------------------------------
