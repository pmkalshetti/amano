function [b,bc] = endpoint_boundary_conditions(V,F,C,E)
    m = size(E, 1);
    c = size(C, 1);
    n = size(V, 1);
    
    % compute closest vertex for each control vertex
    D = pdist2(V, C);
    [minD, Cv] = min(D);
    
    % NaN means no boundary conditions
    bc = repmat(NaN, [n m]);
    
    sqr_d_tol = 1e-6;
    h = avgedge(V,F);  % average of every edge in the mesh
    
    for id_finger = 1:5
        mcp_id = (id_finger-1) * 4 + 2;
        pip_id = mcp_id + 1;
        dip_id = pip_id + 1;
        tip_id = dip_id + 1;
        
        [t_root_mcp,sqr_d] = project_to_lines(V,C(1,:),C(mcp_id,:));
        on_root_mcp = ((abs(sqr_d) < h*sqr_d_tol) & ((t_root_mcp > -1e-10) & (t_root_mcp < (1+1e-10))));
        [t_mcp_pip,sqr_d] = project_to_lines(V,C(mcp_id,:),C(pip_id,:));
        on_mcp_pip = ((abs(sqr_d) < h*sqr_d_tol) & ((t_mcp_pip > -1e-10) & (t_mcp_pip < (1+1e-10))));
        [t_pip_dip,sqr_d] = project_to_lines(V,C(pip_id,:),C(dip_id,:));
        on_pip_dip = ((abs(sqr_d) < h*sqr_d_tol) & ((t_pip_dip > -1e-10) & (t_pip_dip < (1+1e-10))));
        [t_dip_tip,sqr_d] = project_to_lines(V,C(dip_id,:),C(tip_id,:));
        on_dip_tip = ((abs(sqr_d) < h*sqr_d_tol) & ((t_dip_tip > -1e-10) & (t_dip_tip < (1+1e-10))));
        
        
        % root -> mcp
        bone_id = (id_finger-1) * 4 + 1;
        bc(Cv(1), bone_id) = 0;
        bc(on_root_mcp, bone_id) = t_root_mcp(on_root_mcp);
        bc(Cv(mcp_id), bone_id) = 1;
        bc(on_mcp_pip, bone_id) = 1;
        bc(Cv(pip_id), bone_id) = 1;
        bc(on_pip_dip, bone_id) = 1;
        bc(Cv(dip_id), bone_id) = 1;
        bc(on_dip_tip, bone_id) = 1;
        bc(Cv(tip_id), bone_id) = 1;
        
        % mcp -> pip
        bone_id = bone_id + 1;
        bc(Cv(mcp_id), bone_id) = 0;
        bc(on_mcp_pip, bone_id) = t_mcp_pip(on_mcp_pip);
        bc(Cv(pip_id), bone_id) = 1;
        bc(on_pip_dip, bone_id) = 1;
        bc(Cv(dip_id), bone_id) = 1;
        bc(on_dip_tip, bone_id) = 1;
        bc(Cv(tip_id), bone_id) = 1;
        
        % pip -> dip
        bone_id = bone_id + 1;
        bc(Cv(pip_id), bone_id) = 0;
        bc(on_pip_dip, bone_id) = t_pip_dip(on_pip_dip);
        bc(Cv(dip_id), bone_id) = 1;
        bc(on_dip_tip, bone_id) = 1;
        bc(Cv(tip_id), bone_id) = 1;
        
        % dip -> tip
        bone_id = bone_id + 1;
        bc(Cv(dip_id), bone_id) = 0;
        bc(on_dip_tip, bone_id) = t_dip_tip(on_dip_tip);
        bc(Cv(tip_id), bone_id) = 1;
        
    end
    

    
    indices = 1:n;
    % boundary is only those vertices corresponding to rows with at least one non
    % NaN entry
    b = indices(any(~isnan(bc),2));
    bc = bc(b,:);
    % replace NaNs with zeros
    bc(isnan(bc)) = 0;
    
end