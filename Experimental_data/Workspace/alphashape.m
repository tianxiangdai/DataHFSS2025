clear all 
close all
clc
%%
clc
for i=0:2
    r_OP = ncread(strcat("workspace_samples_0_50_step_11_load_", num2str(i), ".nc"), "r_OP")';
    
    
    % get unique points
    tolerance = 1e-12;
    
    unique_mask = true(size(r_OP, 1), 1);
    for j = 1:size(r_OP,1)
        if ~unique_mask(j)
            continue
        end
        dists = sqrt(sum((r_OP - r_OP(j,:)).^2, 2));
        unique_mask(dists < tolerance) = false;
        unique_mask(j) = true;
    end
    r_OP_unique = r_OP(unique_mask, :);
    writematrix(unique_mask, strcat('unique_mask_0_50_step_11_load_', num2str(i), '.csv'))
    min_val = min(r_OP_unique);
    max_val = max(r_OP_unique);
    
    
    shp = alphaShape((r_OP_unique-min_val)./(max_val - min_val), 0.2);
    bf = boundaryFacets(shp);
    writematrix(bf-1, strcat('triangles_0_50_step_11_load_', num2str(i), '.csv'))

    shp.Points = shp.Points.*(max_val - min_val) + min_val;    
    figure(i+1)
    % subplot(1,2,1)
    % plot3(shp.Points(:,1), shp.Points(:,2), shp.Points(:,3),'.')
    % 
    % subplot(1,2,2)
    hold on 
    % plot3(r_OP_bound(:,1),r_OP_bound(:,2),r_OP_bound(:,3),'.r')
    trisurf(bf,r_OP_unique(:,1),r_OP_unique(:,2),r_OP_unique(:,3),...
        'FaceAlpha',0.3, 'EdgeColor', 'none')
%     plot(shp, 'FaceAlpha', 0.3)
    view(0,20)
    hold off
end
%%
for i = 0:0
    disp(i)
    r_OP = ncread(strcat("workspace_samples_0_50_step_11_load_", num2str(i), ".nc"), "r_OP")';
    r_OP = r_OP(end:-1:1,:);
%     r_OP = unique(r_OP', 'rows')';
    
    
    min_val = min(r_OP);
    max_val = max(r_OP);
    
    shp = alphaShape((r_OP-min_val)./(max_val - min_val), 0.2, 'HoleThreshold', 0,'RegionThreshold', 0);
    
    tri = boundaryFacets(shp);
    r_OP_bound = r_OP(unique(tri),:);

    figure(i+1)
%     subplot(1,2,1)
%     plot3(shp.Points(:,1), shp.Points(:,2), shp.Points(:,3),'.')
%     
%     subplot(1,2,2)
%     hold on 
    plot3(r_OP_bound(:,1),r_OP_bound(:,2),r_OP_bound(:,3),'.r')
    trisurf(tri,r_OP(:,1),r_OP(:,2),r_OP(:,3),...
        'FaceColor','cyan','FaceAlpha',0.3, 'EdgeColor', 'b')
    axis equal
    hold off
%     writematrix(tri, strcat('r_OP_bound_surf_0_50_step_11_load_', num2str(i), '.csv'))
end
