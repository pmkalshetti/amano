%% add gptoolbox to path

%% read mesh and skeleton
path_to_obj = './output/hand_model/lbs_weights/mesh_hole_closed.obj';
path_to_tgf = './output/hand_model/lbs_weights/skeleton.tgf';
root_dir = '../../..';
path_to_obj = fullfile(root_dir, path_to_obj);
path_to_tgf = fullfile(root_dir, path_to_tgf);

[V, F] = readOBJ(path_to_obj);
% tsurf(F,V);
[C,E,P,BE,CE,PE] = readTGF(path_to_tgf);
% scatter3(C(:, 1), C(:, 2), C(:, 3));

%% tetrahedralize
S = sample_edges(C, E, 10);
[VV, TT, FF] = tetgen([V;S], F);

%% compute bone weights
% define boundary conditions
[b,bc] = boundary_conditions(VV,TT,C,P,BE,CE);

% bbw
W_per_VV = biharmonic_bounded(VV,TT,b,bc);


W_per_VV = W_per_VV./repmat(sum(W_per_VV,2),1,size(W_per_VV,2)); % weights must sum to 1 for each vertex
W_per_V = W_per_VV(1:size(V,1), :);  % extract weights for surface vertices
W_bone = W_per_V;

%% compute endpoint weights
% define boundary conditions
[b,bc] = endpoint_boundary_conditions(VV,TT,C,BE);

% bbw
W_per_VV = biharmonic_bounded(VV,TT,b,bc);

W_per_V = W_per_VV(1:size(V,1), :);  % extract weights for surface vertices
W_endpoint = W_per_V;

%% save weights
out_path = './output/hand_model/lbs_weights/W.mat';
out_path = fullfile(root_dir, out_path);
save(out_path, 'W_bone', 'W_endpoint');